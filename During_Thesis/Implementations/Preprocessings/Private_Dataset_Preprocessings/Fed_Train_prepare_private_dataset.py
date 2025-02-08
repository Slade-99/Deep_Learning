import os
import random
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from PIL import Image
import cv2

# Paths and parameters
train_dir = '/home/azwad/Works/Deep_Learning/dataset/Private/Train'
train_ratio = 0.9
num_clients = 10
batch_size = 16

# Custom Transforms
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)


# Transform pipeline
new_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    #transforms.RandomRotation(degrees=30),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    #transforms.ToTensor(),
])


eval_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((224, 224)),
    #transforms.ToTensor(),
])


# Load dataset
dataset = datasets.ImageFolder(train_dir, transform=new_transforms)

train_data = dataset


# Redistribute training data among clients
client_data = []
indices = list(range(len(train_data)))
random.shuffle(indices)
split_sizes = np.random.dirichlet(np.ones(num_clients), size=1)[0] * len(train_data)
split_sizes = np.round(split_sizes).astype(int)

start_idx = 0
for i in range(num_clients):
    end_idx = start_idx + split_sizes[i]
    client_indices = indices[start_idx:end_idx]
    client_subset = [train_data[idx] for idx in client_indices]
    client_data.append(client_subset)
    start_idx = end_idx

# Create DataLoaders for each client
client_dataloaders = [
    DataLoader(client_subset, batch_size=batch_size, shuffle=True)
    for client_subset in client_data
]


# Create DataLoader for testing data
#test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Verify distribution

for i, loader in enumerate(client_dataloaders):
    print(f"Client {i + 1} has {len(loader.dataset)} samples.")
#print(f"Test set has {len(test_dataloader.dataset)} samples.")



