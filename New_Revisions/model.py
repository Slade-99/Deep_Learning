import torch
import torch.nn as nn
from inv import involution
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
class involution(nn.Module):

    def __init__(self,channels,kernel_size,stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=channels // reduction_ratio,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels // reduction_ratio,out_channels=kernel_size**2 * self.groups,kernel_size=1,stride=1,)
  
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv1(x if self.stride == 1 else self.avgpool(x))
        weight = self.bn1(weight)
        weight = self.relu(weight)
        weight = self.conv2(weight)
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out
"""

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
            block, layers[1], intermediate_channels=128, stride=1 , type = 'inv'
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2 , type='conv'
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
                block(self.in_channels, intermediate_channels, identity_downsample, stride=2,type=type)
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
    return INN_DWConv(block, [3, 3, 4], img_channel, num_classes)





model = Stage_1_2(1,3)
stage_1_2 = model.to(device)

summary(stage_1_2,(1,224,224))