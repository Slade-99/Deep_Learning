import torch
import torch.nn as nn
from Implementation_Phase.Ablation_Study.Model_Variants.inv import involution
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1 , type='inv'):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels,intermediate_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        if(type=="inv"):
            self.conv2 = involution(intermediate_channels,kernel_size=7,stride=stride)
        
        else:
            self.conv2 = nn.Conv2d(in_channels=intermediate_channels,out_channels=intermediate_channels,kernel_size=(5, 5),stride=stride,padding = 2,groups=intermediate_channels)
        
        
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels,intermediate_channels * self.expansion,kernel_size=1,stride=1,padding=0,bias=False,)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.gelu = nn.GELU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            
            identity = self.identity_downsample(identity)


        x += identity
        x = self.gelu(x)
        return x


class INN_DWConv(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(INN_DWConv, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            1024, 64, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)

        
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1 , type='inv'
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=1 , type = 'conv'
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2 , type='inv'
        )




    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)


        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride,type):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )


        if(type=='conv'):

            
            layers.append(
                block(self.in_channels, intermediate_channels, identity_downsample, stride,type=type)
            )
        else:
            

            layers.append(
                block(self.in_channels, intermediate_channels, identity_downsample, stride,type=type)
            )
            
        
        self.in_channels = intermediate_channels * 4


        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels,type=type))

        return nn.Sequential(*layers)


def Stage_1_2(img_channel=1, num_classes=3):
    return INN_DWConv(block, [2, 2, 2], img_channel, num_classes)





model = Stage_1_2(3,3)
stage_1_2 = model.to(device)

















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
        self.add_module('depthwise_conv', nn.Conv2d(channels, channels, padding=2, kernel_size=5, groups=channels, bias=False))
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
    def __init__(self, channels, blocks, heads, r=2, num_classes=8 ):
        super().__init__()
        self.use_gradCAM = False
        self.gradients = None
        l = []
        out_channels = 256

        l.append(Residual(ConditionalPositionalEncoding(channels)))
        l.append(Residual(LocalAggModule(channels)))
        l.append(Residual(MLP(channels)))
        l.append(Residual(ConditionalPositionalEncoding(channels)))
        l.append(Residual(GlobalSparseAttetionModule(channels=channels, r=r, heads=heads)))
        l.append(Residual(MLP(channels)))

        self.dropout_attention = nn.Dropout(0.4)
        self.dropout_fc = nn.Dropout(0.5)
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
        x = self.dropout_attention(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        
        if self.use_gradCAM:
            h = x.register_hook(self.activations_hook)
        x = self.pooling(x).flatten(1)

        x = self.dropout_fc(x)
        x = self.classifier(x)
        
        return x


    def get_activations_gradient(self):
        return self.gradients
    

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








invo_sparse_net = prepare_architecture().to(device)
#print(model)
#summary(invo_sparse_net, input_size =(1,224,224))
#x = torch.rand(16,1,224,224).to(device)
#print(invo_sparse_net(x).shape)