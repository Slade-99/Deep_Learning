import torch
import torch.nn as nn
from involution import Involution2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1 , type='inv' 
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        self.dropout = nn.Dropout(0.2)
        
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        if(type=="inv"):
            self.conv2 = Involution2d(
                intermediate_channels,
                intermediate_channels,
                kernel_size=(7,7),
                stride=(stride,stride),
                groups = 16,
                reduce_ratio= 4
            )
        else:



            self.conv2 = nn.Conv2d(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=(3, 3),
                stride=(stride,stride),
                padding = 1,
                groups=intermediate_channels  #Depthwise
                )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.GELU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.relu(x)
        if self.training:
            
            x  = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        image_channels = 3
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            1024, 16, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.ConvTranspose2d(16, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1 , type='inv'
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2 , type = 'inv'
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2 , type='conv'
        )




    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.training:
            x  = self.dropout(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x=  self.relu(x)
        if self.training:
            
            x  = self.dropout(x)


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


def ResNet50(img_channel=3, num_classes=10):
    return ResNet(block, [3, 3, 4], img_channel, num_classes)





model = ResNet50(3,10)
custom_resnet = model.to(device)