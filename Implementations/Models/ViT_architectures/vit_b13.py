import matplotlib.pyplot as plt
import torch
import torchvision
import random
import numpy as np
#from transformers import CvTModel, CvTConfig
from torch import nn
from torchvision import transforms
from torchinfo import summary
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        
        # Load the Vision Transformer model without pretrained weights
        self.vit = torchvision.models.vit_b_16(weights=None)
        
        # Modify the input layer to accept 1-channel grayscale images
        # The original input layer expects 3 channels, we replace it with one
        self.vit.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=(16, 16), stride=(16, 16))
        
        # Adjust the classifier head
        self.vit.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)

    def forward(self, x):
        return self.vit(x)

class_names = ['normal', 'covid', 'pneumonia']
# Initialize the ViT model without pretrained weights
pretrained_vit = CustomViT(3).to(device)

# Change the input layer to accept 1 channel (grayscale)
#retrained_vit.patch_embed.proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=(16, 16), stride=(16, 16))

# Change the classifier head





# Print a summary using torchinfo
summary(model=pretrained_vit, 
        input_size=(16, 1, 224, 224),  # 1 channel for grayscale images
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# Define transforms for grayscale images
pretrained_vit_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])


# Setup directory paths to train and test images
train_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/train'
test_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/test'



NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int=NUM_WORKERS):
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=pretrained_vit_transforms,
    batch_size=16
)

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()












#Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    pretrained_vit.train()  # Set the model to training mode
    running_loss = 0.0
    
    # Use tqdm to display progress
    for images, labels in tqdm(train_dataloader_pretrained, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear gradients
        outputs = pretrained_vit(images)  # Get model predictions
        loss = loss_fn(outputs, labels)  # Calculate loss
        
        # Backward pass and optimization
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update parameters
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader_pretrained)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
pretrained_vit.eval()  # Set the model to evaluation mode
all_labels = []
all_preds = []

with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_dataloader_pretrained:
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_vit(images)
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Classification Report
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()