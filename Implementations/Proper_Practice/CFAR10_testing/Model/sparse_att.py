import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







edgevit_configs = {
    'C': {
        'channels': (32,),
        'blocks': (1,),
        'heads': (4,)
    }
}






HYPERPARAMETERS = {
    'r': (4, 2, 2, 1)
}






class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


        self.convtranspose = nn.ConvTranspose2d(32,32, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        # Get the output of the module
        out = self.module(x)

        # If the channels don't match, apply a pointwise (1x1) convolution to the input tensor

            # Apply 1x1 convolution to match the number of channels
        if out.shape[2] != x.shape[2] or out.shape[3] != x.shape[3]:
             x = self.convtranspose(x)



        return x + out



class ConditionalPositionalEncoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('conditional_ositional_encoding', nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False))


class MLP(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        expansion = 4
        self.add_module('mlp_layer_0', nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False))
        self.add_module('mlp_act', nn.GELU())
        self.add_module('mlp_layer_1', nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False))


class LocalAggModule(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('pointwise_prenorm_0', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_0', nn.Conv2d(channels, channels, kernel_size=1, bias=False))
        self.add_module('depthwise_conv', nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False))
        self.add_module('pointwise_prenorm_1', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_1', nn.Conv2d(channels, channels, kernel_size=1, bias=False))


class GlobalSparseAttetionModule(nn.Module):
    def __init__(self, channels, r, heads):
        super().__init__()
        self.head_dim = channels//heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads

        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.local_prop = nn.ConvTranspose2d(channels, channels, kernel_size=r, stride=r, groups=channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H*W).split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)

        return x


class ConvDownsampling(nn.Sequential):
    def __init__(self, inp, oup, r, bias=False):
        super().__init__()
        self.add_module('downsampling_conv', nn.Conv2d(inp, oup, kernel_size=r, stride=r, bias=bias))
        self.add_module('downsampling_norm', nn.GroupNorm(num_groups=1, num_channels=oup))


class EdgeViT(nn.Module):
    def __init__(self, channels, blocks, heads, r=[4, 2, 2, 1], num_classes=10, distillation=False):
        super().__init__()
        self.distillation = distillation

        l = []
        in_channels = 256
        for stage_id, (num_channels, num_blocks, num_heads, sample_ratio) in enumerate(zip(channels, blocks, heads, r)):
            l.append(ConvDownsampling(inp=in_channels, oup=num_channels, r=4 if stage_id == 0 else 2))

            for _ in range(num_blocks):
                l.append(Residual(ConditionalPositionalEncoding(num_channels)))
                l.append(Residual(LocalAggModule(num_channels)))
                l.append(Residual(MLP(num_channels)))
                l.append(Residual(ConditionalPositionalEncoding(num_channels)))
                l.append(Residual(GlobalSparseAttetionModule(channels=num_channels, r=sample_ratio, heads=num_heads)))
                l.append(Residual(MLP(num_channels)))

            in_channels = num_channels

        self.main_body = nn.Sequential(*l)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(in_channels, num_classes, bias=True)

        if self.distillation:
            self.dist_classifier = nn.Linear(in_channels, num_classes, bias=True)

    def forward(self, x):
        x = self.main_body(x)
        x = self.pooling(x).flatten(1)

        if self.distillation:
            x = self.classifier(x), self.dist_classifier(x)

            if not self.training:
                x = 1/2 * (x[0] + x[1])
        else:
            if self.training:
                
                x = self.dropout(x)
            x = self.classifier(x)

        return x




def EdgeViT_C(pretrained=False):
    model = EdgeViT(**edgevit_configs['C'])

    if pretrained:
        raise NotImplementedError

    return model








sparse_attn_model = EdgeViT_C(False)
sparse_attn_model = sparse_attn_model.to(device)