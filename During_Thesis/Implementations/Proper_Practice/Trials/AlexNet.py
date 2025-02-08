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
batch_size = 16
num_epochs = 1
# Load Data






# AlexNET
class AlexNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(AlexNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=0,
        )
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        
        
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
        )


        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.fc6 = nn.Linear(9216, 4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096, num_classes)





    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






train_loader = train_dataloader
test_loader = test_dataloader








# Initialize network
model = AlexNet(in_channels=in_channels, num_classes=num_classes).to(device)

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