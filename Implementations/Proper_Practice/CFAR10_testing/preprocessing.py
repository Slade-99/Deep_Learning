
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import os



class_names = ['normal', 'pneumonia', 'abnormal' ]
train_dir = '/home/azwad/Works/Deep_Learning/dataset/unified/'




####### Dataset Preparation ########
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img)

class MedianBlurTransform:
    def __call__(self, img):
        img = np.array(img)
        img = cv2.medianBlur(img, 5)
        return Image.fromarray(img)



new_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    #MedianBlurTransform(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])




train_data = datasets.ImageFolder(train_dir, transform=new_transforms)

class_names = train_data.classes







def show_images_from_dataloader(dataloader, class_names, num_images=1):
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
#show_images_from_dataloader(train_dataloader, class_names)


