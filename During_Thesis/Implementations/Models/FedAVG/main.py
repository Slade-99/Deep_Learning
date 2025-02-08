# Required Libraries
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Data Directories
train_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/train/'
test_dir = '/home/azwad/Datasets/Benchmark_Dataset/Data/test/'

# Data Preparation
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ]),
    transforms.ToTensor(),
])

# Load Data
train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    for image in os.listdir(train_dir + label):
        train_paths.append(train_dir + label + '/' + image)
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# Encode Labels
unique_labels = list(set(train_labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
encoded_labels = [label_to_index[label] for label in train_labels]

# Train-Test Split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    train_paths, encoded_labels, test_size=0.2, random_state=42)

# Create DataLoaders for Test Set
batch_size = 32
test_dataset = CustomImageDataset(test_paths, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Model
class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.base_model = vgg16(weights='DEFAULT').features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512 * 4 * 4, 128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = self.classifier(x)
        return x

# Initialize Model
model = CustomVGG16(num_classes=len(unique_labels))

# Federated Learning Settings
NUM_CLIENTS = 2
NUM_ROUNDS = 2

# Split dataset among clients
clients = []
for i in range(NUM_CLIENTS):
    client_data = train_paths[i * (len(train_paths) // NUM_CLIENTS):(i + 1) * (len(train_paths) // NUM_CLIENTS)]
    client_labels = train_labels[i * (len(train_labels) // NUM_CLIENTS):(i + 1) * (len(train_labels) // NUM_CLIENTS)]
    clients.append((client_data, client_labels))

# Create DataLoaders for each client
client_datasets = []
for client in clients:
    client_datasets.append(CustomImageDataset(client[0], client[1], transform=transform))

# Federated Learning Loop
criterion = nn.CrossEntropyLoss()

for round_num in range(NUM_ROUNDS):
    print(f"Starting round {round_num + 1}")
    selected_clients = random.sample(client_datasets, int(NUM_CLIENTS * 0.5))
    client_weights = []

    # Transmit the global model to the selected clients
    for client_dataset in selected_clients:
        client_model = CustomVGG16(num_classes=len(unique_labels))
        client_model.load_state_dict(model.state_dict())

        client_optimizer = optim.Adam(client_model.parameters(), lr=0.0001)

        # Train locally
        client_model.train()
        for epoch in range(2):
            print(f"  Training on client for epoch {epoch + 1}")
            for images, labels in DataLoader(client_dataset, batch_size=batch_size, shuffle=True):
                client_optimizer.zero_grad()
                outputs = client_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                client_optimizer.step()
                print("step taken")

        # Store the client's weights
        client_weights.append(client_model.state_dict())
    
    print("Done with local training for selected clients")

    # Aggregate the model weights
    new_weights = {}
    for key in client_weights[0].keys():
        new_weights[key] = torch.mean(torch.stack([client_weights[i][key] for i in range(len(client_weights))]), dim=0)

    # Update the global model with the aggregated weights
    model.load_state_dict(new_weights)

# Evaluate the global model
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Print Classification Report
print(classification_report(y_true, y_pred, target_names=unique_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=unique_labels, yticklabels=unique_labels, annot_kws={"fontsize": 20}, cbar=False)
plt.xlabel("Predicted Label", fontsize=20)
plt.ylabel("True Label", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20, rotation=0)
plt.show()

# Save the model
torch.save(model.state_dict(), 'my_model.pth')
