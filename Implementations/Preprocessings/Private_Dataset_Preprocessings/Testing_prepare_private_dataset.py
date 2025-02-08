import os
import random
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from PIL import Image
import cv2

# Paths and parameters
train_dir = '/home/azwad/Works/Deep_Learning/dataset/Private/Test'
num_clients = 10
batch_size = 16

class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)


eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load dataset
dataset = datasets.ImageFolder(train_dir, transform=eval_transforms)

test_data = dataset




# Create DataLoader for testing data
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


