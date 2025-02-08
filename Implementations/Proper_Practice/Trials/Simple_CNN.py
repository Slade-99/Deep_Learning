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
in_channels = 1
num_classes = 3
learning_rate = 3e-4 # karpathy's constant
batch_size = 8
num_epochs = 30
# Load Data

def save_checkpoint(state, filename="/home/azwad/Works/DL_Models_Checkpoint/LeNET.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])





# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 56 * 56, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






train_loader = train_dataloader
test_loader = test_dataloader




def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            loss = criterion(scores, y)
            total_loss += loss.item()

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples
    average_loss = total_loss / len(loader)

    return accuracy , average_loss



# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



### Load checkpoint ####
#load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


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

    accuracy , loss = check_accuracy(train_loader, model)
    accuracy = int((accuracy*100).item())
    accuracy = round(accuracy, 2)
    loss = round((loss*100),2)
    accuracy_2 , loss_2 = check_accuracy(test_loader, model)
    accuracy_2 = int((accuracy_2*100).item())
    accuracy_2 = round(accuracy_2, 2)
    loss_2 = round((loss_2*100),2)

    print(f"%(Accuracy,Loss) on training set at epoch->{epoch+1}: {(accuracy)}\n")
    print(f"%(Accuracy,Loss) on test set at epoch->{epoch+1}: {(accuracy_2)}\n")







