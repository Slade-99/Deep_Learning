# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules  # Gives easier dataset managment by creating mini batches etc.
from data_loader import train_dataloader, test_dataloader
from tqdm import tqdm  # For nice progress bar!



# Hyperparameters
in_channels = 1
num_classes = 3
learning_rate = 3e-4 # karpathy's constant
batch_size = 4
num_epochs = 5
# Load Data

def save_checkpoint(state, filename="/home/azwad/Works/DL_Models_Checkpoint/MobileNetV1.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])




# MobileNetV1
class MobileNetV1(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(MobileNetV1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        
        
        self.conv1 = nn.Conv2d(1,32,3,2,1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        self.dwsc1 = depthwise_conv_block(32,64,stride=1,padding=1)
        
        
        self.pooling = nn.AvgPool2d(kernel_size=(7,7))
        self.fc = nn.Linear(1024,self.num_classes)
        
        

        
        


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x
    
    
    
class depthwise_conv_block(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(depthwise_conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,3), *kwargs ,groups=in_channels)
        self.batchnorm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
                

class pointwise_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(pointwise_conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1) , stride=1, padding=0 )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
                

class depthwise_seperable_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,stride, padding):
        super(depthwise_seperable_conv_block, self).__init__()
        self.depthwise_layer = depthwise_conv_block(in_channels,stride = stride , padding = padding)
        self.pointwise_layer = pointwise_conv_block(in_channels,out_channels)

    def forward(self, x):
        return self.pointwise_layer(self.depthwise_layer(x))





class Inception_block(nn.Module):
    def __init__(self, in_channels,out_1x1, red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1pool):
        super(Inception_block,self).__init__()
        
        self.branch1 =  conv_block(in_channels,out_1x1,kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=1),
        )
        
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )
        
        
    def forward(self,x):  ## Remember the format : batch_size, channels , width , height 
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




















# Initialize network
model = MobileNetV1(in_channels=in_channels, num_classes=num_classes).to(device)

x = torch.rand(1,1,224,224)
print(model(x).shape)



"""

train_loader = train_dataloader
test_loader = test_dataloader
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



## Load if checkpoint Available ###


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

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)





# Check accuracy on training & test to see how good our model
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


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

"""