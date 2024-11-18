import torch
import torch.nn as nn
import torch
import time
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules  # Gives easier dataset managment by creating mini batches etc.
from data_loader import train_dataloader, test_dataloader
from tqdm import tqdm  # For nice progress bar!




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channels = 1
num_classes = 3
learning_rate = 0.001 
batch_size = 16
num_epochs = 5


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        
        
        self.conv1 =nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.inv1 = Involution(out_channels,out_channels,(7,7),padding=(3,3),stride=(1,1),groups=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.inv1(x)
        x = self.bn2(x)


        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)


        x += identity
        x = self.relu(x)
        return x
    
    


class RedNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(RedNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.inv1 = Involution(32,32,(3,3),padding=(1,1),stride=(1,1),groups=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    
        self.layer1 = self._make_layer(
            block, layers[0], out_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], out_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], out_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], out_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    





    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []


        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels ,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels ),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )

        
        self.in_channels = out_channels


        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    

class Involution(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, groups, reduce_ratio=1, dilation=(1, 1), padding=(3, 3), bias=False):
        super().__init__()
        self.bias = bias
        self.padding = padding
        self.dilation = dilation
        self.reduce_ratio = reduce_ratio
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.output_ch = output_ch
        self.input_ch = input_ch
        self.init_mapping = nn.Conv2d(in_channels=self.input_ch, out_channels=self.output_ch, kernel_size=(1, 1), stride=(1, 1), bias=self.bias) if self.input_ch != self.output_ch else nn.Identity()
        self.reduce_mapping = nn.Conv2d(in_channels=self.input_ch, out_channels=self.output_ch // self.reduce_ratio, kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.span_mapping = nn.Conv2d(in_channels=self.output_ch // self.reduce_ratio, out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups, kernel_size=(1, 1), stride=(1, 1),
                                      bias=self.bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        self.pooling = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.sigma = nn.Sequential(
            nn.BatchNorm2d(num_features=self.output_ch // self.reduce_ratio, momentum=0.3), nn.ReLU())

    def forward(self, inputs):
        batch_size, _, in_height, in_width = inputs.shape
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                     // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                    // self.stride[1] + 1

        unfolded_inputs = self.unfold(self.init_mapping(inputs))
        inputs = F.adaptive_avg_pool2d(inputs,(out_height,out_width))
        unfolded_inputs = unfolded_inputs.view(batch_size, self.groups, self.output_ch // self.groups, self.kernel_size[0] * self.kernel_size[1], out_height, out_width)

        kernel = self.pooling(self.span_mapping(self.sigma(self.reduce_mapping((inputs)))))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        output = (kernel * unfolded_inputs).sum(dim=3)

        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
        return output
    




model = RedNet(block, [ 2,2,2,2], 1 ,3).to(device)



x = torch.rand(1,1,224,224)



train_loader = train_dataloader
test_loader = test_dataloader
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



## Load if checkpoint Available ###


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples



print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")




# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    
        time.sleep(5)

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    #save_checkpoint(checkpoint)
    
    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")




# Check accuracy on training & test to see how good our model

