import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import torchvision.datasets as datasets 
import os
from Implementation_Phase.Models.InvoSparseNet.model_pca import invo_sparse_net
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
dataset = "Private_CXR"


data_dir = "/mnt/hdd/Datasets/" + dataset + "/"
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
# Load datasets
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=data_transforms["val"])
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=data_transforms["test"])
class_names = train_dataset.classes
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




model = invo_sparse_net
model.eval()
learning_rate = 0.000005
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
def load_checkpoint(checkpoint,optimizer):
  print("=>Loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  #optimizer.load_state_dict(checkpoint['optimizer'])
load = True
if load:
    model_path = '/mnt/hdd/Trained_Weights/Private_CXR/invo_sparse_net/invo_sparse_net_1740581115.7178764.pth.tar'
    checkpoint = torch.load(model_path)
    load_checkpoint(checkpoint,optimizer)

# Extract features from the dataset
features_list = []
labels_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        _, features = model(images, return_features=True)  # Get feature activations
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# Convert to NumPy arrays
features = np.concatenate(features_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2D for visualization
reduced_features = pca.fit_transform(features)

# Visualize PCA result
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='jet', alpha=1)
plt.colorbar(scatter, label="Class Labels").ax.set_ylabel("Class Labels", fontsize=14, fontweight="bold")
plt.xlabel("Principal Component 1" , fontsize=14, fontweight="bold" )
plt.ylabel("Principal Component 2" , fontsize=14, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")

#plt.title("PCA Analysis of Model Feature Representations")
plt.show()
