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
from transformers import CvtModel, CvtConfig
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomCvT(nn.Module):
    def __init__(self , num_labels):
        super(CustomCvT, self).__init__()

        # Load the CvT model configuration
        config = CvtConfig(num_channels=1 , num_labels = 3)
        self.cvt = CvtModel(config ,add_pooling_layer = True )
        self.classifier = nn.Linear(384, num_labels)


    def forward(self, x):
        output = self.cvt(x)
        cls_token_output = output.cls_token_value  # Get the CLS token output
        cls_token_output = cls_token_output.squeeze(1)  # Remove the second dimension to get shape [batch_size, hidden_size]
          # Print the shape
        logits = self.classifier(cls_token_output)  # Pass through the classification head
        return logits
        
    

class_names = ['normal', 'covid', 'pneumonia']
# Initialize the CvT model
pretrained_cvt = CustomCvT(3).to(device)
summary(pretrained_cvt, input_size=(1, 1,224, 224)) 

# Print a summary using torchinfo
"""
summary(model=pretrained_cvt, 
        input_size=(16, 1, 224, 224),  # 1 channel for grayscale images
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)



pretrained_cvt_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])



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
    transform=pretrained_cvt_transforms,
    batch_size=16
)

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_cvt.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()



num_epochs = 1
for epoch in range(num_epochs):
    pretrained_cvt.train()  # Set the model to training mode
    running_loss = 0.0
    
    # Use tqdm to display progress
    for images, labels in tqdm(train_dataloader_pretrained, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear gradients
        outputs = pretrained_cvt(images)  # Get model predictions
        loss = loss_fn(outputs, labels)  # Calculate loss
        
        # Backward pass and optimization
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update parameters
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader_pretrained)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
pretrained_cvt.eval()  # Set the model to evaluation mode
all_labels = []
all_preds = []

with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_dataloader_pretrained:
        images, labels = images.to(device), labels.to(device)
        outputs = pretrained_cvt(images)
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
"""