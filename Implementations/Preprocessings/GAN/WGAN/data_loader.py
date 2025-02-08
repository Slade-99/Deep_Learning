
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import os
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter




class_names = ['normal', 'pneumonia','abnormal']
train_dir = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/Fresh/train'
test_dir = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/Fresh/test'


####### Dataset Preparation ########
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(7, 7))
        img = clahe.apply(img)
        return Image.fromarray(img)

class MedianBlurTransform:
    def __call__(self, img):
        img = np.array(img)
        img = cv2.medianBlur(img, 5)
        return Image.fromarray(img)



new_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    #CLAHETransform(),
    #MedianBlurTransform(),
    #transforms.RandomRotation(degrees=15),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])

new_transforms_2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    #MedianBlurTransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=new_transforms_2)
    class_names = train_data.classes
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,


    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,


    )
    return train_dataloader, test_dataloader, class_names

train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=new_transforms,
    batch_size=8
)









def show_images_from_dataloader(dataloader, class_names, num_images=8):
    # Get a batch of images from the dataloader
    images, labels = next(iter(dataloader))

    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images[:num_images], nrow=4, padding=2)

    # Convert from Tensor to NumPy for displaying with matplotlib
    np_img = img_grid.numpy()

    # Plot the images
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis("off")
    plt.title("Sample Images")
    plt.show()

    # Print class names for the selected images
    print("Labels:", [class_names[label] for label in labels[:num_images]])

# Display images from the training dataloader
show_images_from_dataloader(train_dataloader, class_names)
show_images_from_dataloader(test_dataloader, class_names)