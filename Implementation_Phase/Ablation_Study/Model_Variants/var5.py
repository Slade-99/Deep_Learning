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
            self.conv2 = nn.Conv2d(in_channels=intermediate_channels,out_channels=intermediate_channels,kernel_size=(3, 3),stride=stride,padding = 1,groups=intermediate_channels)
        
        
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





model = Stage_1_2(1,3)
stage_1_2 = model.to(device)





class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,num_classes=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim*196,num_classes)

    def forward(self, x):
        B , X = x.shape
        x = x.view(B,-1,self.embed_dim)
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / self.head_dim ** 0.5, dim=-1)
        out = (attn_weights @ v).reshape(batch_size, seq_length, embed_dim)
        out = self.out_proj(out).flatten(1)
        out = self.classifier(out)
        return out



class AxialSelfAttention(nn.Module):
    def __init__(self, embed_dim,num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.scale = embed_dim ** 0.5
        self.classifier = nn.Linear(embed_dim,num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B,-1,self.embed_dim)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
       
        # Row-wise attention
        attn_row = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        out_row = attn_row @ v

        # Column-wise attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_col = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        out_col = attn_col @ v

        return out_row + out_col.transpose(1, 2)






attn = AxialSelfAttention(256,3)











class Custom_Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_1_2 = stage_1_2
        self.attn = attn
        self.classifier = nn.Linear(196*256,3)
        

    
    def forward(self, x):
        x = self.stage_1_2(x)
        
        
        x = self.attn(x)
        x = x.flatten(1)
        x= self.classifier(x)
        return x



    








def prepare_architecture():
    model = Custom_Architecture()

    return model








invo_sparse_net = prepare_architecture().to(device)
#print(model)
#summary(invo_sparse_net, input_size =(1,224,224))
x = torch.rand(16,1,224,224).to(device)
#print(invo_sparse_net(x).shape)