# Imports
import torch
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim, nn  
from torchvision.models import mobilenet_v3_small,mobilenet_v2
from New_Revisions.model import model
from torch.utils.data import DataLoader
from tqdm import tqdm  
from torchsummary import summary
import time
import os
import numpy as np
from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
in_channels = 1  
num_classes = 3
learning_rate = 0.0001
batch_size = 16
num_epochs = 100



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







###### Model Preparation #######
"""
model = mobilenet_v2(num_classes=3)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model = model.to(device)
"""
model = model.to(device)
################################


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




# Train Network
def train(model):
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
            if(batch_idx%100 == 0):
                time.sleep(5)
        time.sleep(10)
        print(f"Results on epoch {epoch+1}")
        print("------------------------------")
        print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
        print(f"Accuracy on validation set: {check_accuracy(val_loader, model) * 100:.2f}%")
        print("\n\n")




    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
    
    
train(model)