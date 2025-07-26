from PIL import Image
import os
import torchvision.transforms as transforms  
import torchvision.datasets as datasets 

dataset = "NIH"
data_dir = "/mnt/hdd/Datasets/" + dataset + "/"



# Data transformations with augmentation for training
data_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale

        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    "test": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}



# Load datasets
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=data_transforms["train"])

for path, _ in train_dataset.samples:
    try:
        img = Image.open(path)
        img.verify()  # Checks for corruption
    except Exception as e:
        print(f"Corrupt image: {path} -> {e}")
