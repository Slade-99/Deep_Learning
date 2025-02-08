# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules  # Gives easier dataset managment by creating mini batches etc.
from data_loader import train_dataloader, test_dataloader
from tqdm import tqdm  # For nice progress bar!
import time


# Hyperparameters
VGG_16 = [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]
in_channels = 1
num_classes = 4
learning_rate = 3e-4 # karpathy's constant
batch_size = 4
num_epochs = 5
# Load Data

def save_checkpoint(state, filename="/home/azwad/Works/DL_Models_Checkpoint/VGG16.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])




class VGG16(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(VGG_16)
        
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        
        


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        return x
    
    
    
    def create_conv_layers(self,architecture):
        layers = [ ]
        
        in_channels = self.in_channels
        
        
        for layer in architecture:
            
            if type(layer) == int:
                
                out_channels = layer
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1)) ,
                            nn.BatchNorm2d(layer), 
                            nn.ReLU()]
                
                in_channels = layer

            else:
                
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                
        return nn.Sequential(*layers)
                
        



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




















# Initialize network
model = VGG16(in_channels=in_channels, num_classes=num_classes).to(device)

x = torch.rand(1,1,224,224)
print(model(x).shape)





train_loader = train_dataloader
test_loader = test_dataloader
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



## Load if checkpoint Available ###

"""
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
        
        time.sleep(3)

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)
"""


load_checkpoint(torch.load("/home/azwad/Works/DL_Models_Checkpoint/VGG16.pth.tar"))

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

