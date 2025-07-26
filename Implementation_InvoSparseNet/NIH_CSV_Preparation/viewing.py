import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import cv2
from PIL import Image
dataset = "NIH"
# Assuming your code snippet is already executed and the following variables are defined:
# data_dir, dataset, batch_size, data_transforms, train_loader
data_dir = "/mnt/hdd/Datasets/" + dataset + "/"
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)
data_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        CLAHETransform(),
        transforms.Resize((224, 224)),
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
batch_size = 16
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=data_transforms["val"])
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=data_transforms["test"])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
def imshow(img):
    """Function to show a grayscale image."""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))[...,0], cmap='gray') #added cmap='gray'
    plt.show()

# Get a random batch from the train loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Get a single random image from the batch
random_index = np.random.randint(0, images.size(0))  # Get a random index within the batch
random_image = images[random_index]

# Display the image
imshow(random_image)

# Optionally, you can also print the label of the image
print(f"Label: {labels[random_index].item()}")