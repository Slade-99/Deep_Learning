import torch
import torch.nn as nn
from Implementations.Proper_Practice.Final_Testing.Model.Custom_Architecture.inn_dwconv import stage_1_2
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







architecture_configs = {
    'C': {
        'channels': (64,),
        'blocks': (1,),
        'heads': (4,)
    }
}






HYPERPARAMETERS = {
    'r': (2)
}






class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


        self.convtranspose = nn.ConvTranspose2d(64,64, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        
        out = self.module(x)

        

        
        if out.shape[2] != x.shape[2] or out.shape[3] != x.shape[3]:
             x = self.convtranspose(x)



        return x + out



class ConditionalPositionalEncoding(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('conditional_positional_encoding', nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False))


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
        attn = (((q.transpose(-2, -1) @ k))*self.scale).softmax(-1)
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


class Custom_Architecture(nn.Module):
    def __init__(self, channels, blocks, heads, r=2, num_classes=3 ):
        super().__init__()
        self.use_gradCAM = False
        self.gradients = None
        l = []
        out_channels = 1024

        l.append(Residual(ConditionalPositionalEncoding(channels)))
        l.append(Residual(LocalAggModule(channels)))
        l.append(Residual(MLP(channels)))
        l.append(Residual(ConditionalPositionalEncoding(channels)))
        l.append(Residual(GlobalSparseAttetionModule(channels=channels, r=r, heads=heads)))
        l.append(Residual(MLP(channels)))

            

        self.stage_1_2 = stage_1_2
        self.attn_body = nn.Sequential(*l)
        self.conv1 = nn.Conv2d(64,out_channels,1,1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        # placeholder for the gradients
        self.gradients = None
        self.classifier = nn.Linear(out_channels, num_classes, bias=True)


    
    

    def activations_hook(self, grad):
        self.gradients = grad
        
    
    
    def forward(self, x):
        x = self.stage_1_2(x)
        x = self.attn_body(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        
        if self.use_gradCAM:
            h = x.register_hook(self.activations_hook)
        x = self.pooling(x).flatten(1)


        x = self.classifier(x)

        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        x = self.stage_1_2(x)
        x = self.attn_body(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        return x




def prepare_architecture():
    model = Custom_Architecture(channels=64,blocks=1,heads=4)

    return model








model = prepare_architecture().to(device)
print(model)
summary(model, input_size =(1,224,224))