# Imports
import torch
import torch.nn.functional as F  
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms  
from torch import optim, nn  
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader
from tqdm import tqdm  
import os
import numpy as np
from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
in_channels = 1  
num_classes = 3
learning_rate = 0.001  
batch_size = 64
num_epochs = 50



class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)

# Data transformations with augmentation for training
data_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "test": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}

# Define dataset directory
data_dir = "/mnt/hdd/dataset/Private/Finalized_from_StyleGAN2_ada/Train_Val_Test/"  # Change this to your dataset folder



# Load datasets
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Train"), transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Val"), transform=data_transforms["val"])
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Test"), transform=data_transforms["test"])


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Simple CNN
"""
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, num_classes)  # Adjusted for 64x64 input

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
"""



# Initialize network
model = mobilenet_v3_small(num_classes=3)
model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)  # Change 3 -> 1 channel
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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




for i in range(5):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.cpu()  # Move to CPU if necessary

        # Select a random image from the batch
        img = data[torch.randint(0, data.shape[0], (1,))].squeeze(0)  # Remove batch dimension

        # Convert shape from (1, H, W) → (H, W) for grayscale display
        img = img.squeeze(0)  

        # Plot the image
        plt.imshow(img, cmap='gray')  # Use 'gray' for single-channel images
        plt.axis("off")  # Hide axes
        plt.title("Random Image from Batch")
        plt.show()

        break  # Show only one image



#print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")