import torch
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()  # Remove the last classification layer
resnet.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom dataset class to load images
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)


# Set paths to real and fake image folders
real_images_folder = "/mnt/hdd/dataset/Private/raw/unified/"
fake_images_folder = "/mnt/hdd/Dataset_augmentation/generated_images/unified/"

# Load datasets
real_dataset = ImageFolderDataset(real_images_folder, transform)
fake_dataset = ImageFolderDataset(fake_images_folder, transform)

# Create DataLoaders
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# Function to extract features
def extract_features(model, dataloader):
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.cuda()  # Move to GPU if available
            feat = model(batch).cpu().numpy()  # Extract features
            features.append(feat)
    return np.concatenate(features, axis=0)

# Extract features from ResNet-50
real_features = extract_features(resnet, real_loader)
fake_features = extract_features(resnet, fake_loader)



# Combine real and fake features
all_features = np.vstack((real_features, fake_features))
labels = np.array([0] * len(real_features) + [1] * len(fake_features))  # 0: Real, 1: Fake

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2D for visualization
pca_features = pca.fit_transform(all_features)

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[labels == 0, 0], pca_features[labels == 0, 1], alpha=0.5, label="Real Images", color='blue')
plt.scatter(pca_features[labels == 1, 0], pca_features[labels == 1, 1], alpha=0.5, label="Generated Images", color='red')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.title("PCA Analysis of Real vs Generated Images")
plt.show()
