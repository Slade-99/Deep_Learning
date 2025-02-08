
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision import datasets
import cv2
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Subset

class_names = ['COVID','Normal', 'Pneumonia']
train_dir = '/home/azwad/Works/Deep_Learning/dataset/Benchmark/unified/'




####### Dataset Preparation ########
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)

        # Fix shape: (1, H, W) → (H, W)
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)  # Remove the channel dimension

        # Convert to grayscale if RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            pass  # Already grayscale
        else:
            raise ValueError(f"Invalid image shape after squeeze: {img.shape}")

        # Ensure uint8 data type
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        return Image.fromarray(img)






##### Augmentation Transform (Applied Dynamically) #####
aug_transform = transforms.Compose([
    #CLAHETransform(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    #CLAHETransform(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])


##### Dynamic Augmentation Dataset #####
class AugmentedDataset(Dataset):
    def __init__(self, dataset, augmentations=None):
        self.dataset = dataset
        if(augmentations=='val'):
            self.augmentations = val_transform
        elif(augmentations=='aug'):
            self.augmentations = aug_transform
            
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the image and label from the dataset
        image, label = self.dataset[idx]

        # Apply augmentations (if any) on the image
        if self.augmentations:
            image = self.augmentations(image)

        return image, label

