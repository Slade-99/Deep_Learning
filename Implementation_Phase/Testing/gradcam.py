import torch
import torch.nn as nn
from Implementation_Phase.Models.InvoSparseNet.model import invo_sparse_net
from torchsummary import summary
import numpy as np
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import torchvision.datasets as datasets 
from PIL import Image
import os
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['normal', 'pneumonia', 'abnormal' ]
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
        transforms.ToTensor(),

    ]),
    "val": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CLAHETransform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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














# Preparing the Trained Model
model_path = '/mnt/hdd/Trained_Weights/Private_CXR/invo_sparse_net/invo_sparse_net_1740581115.7178764.pth.tar'
model = invo_sparse_net.to(device)
learning_rate = 0.0001
checkpoint = torch.load(model_path)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

img = None
#train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# Prepare the subplot grid
num_images = 20  # Number of images to display
rows = 4  # Number of rows
cols = 5  # Number of columns
fig, axes = plt.subplots(rows, cols, figsize=(10,10))  # Adjust the size accordingly
axes = axes.flatten()  # Flatten axes array for easier indexing

for idx, (images, labels) in enumerate(test_loader):
    if idx >= 1:
        break  # We only need to process the first batch

    for i in range(min(num_images, images.shape[0])):  # Display up to num_images images from the batch
        single_image = images[i].unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Perform prediction
        pred = model(single_image)
        class_index = pred.argmax(dim=1).item()

        # Compute gradients with respect to the predicted class
        pred[:, class_index].backward()

        # Retrieve gradients and activations
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(single_image).detach()

        # Weight the activations by the pooled gradients
        for j in range(activations.shape[1]):  # Dynamically handle channel count
            activations[:, j, :, :] *= pooled_gradients[j]

        # Compute the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = heatmap / torch.max(heatmap)  # Normalize
        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # Apply ReLU
        heatmap = cv2.resize(heatmap, (single_image.shape[2], single_image.shape[3]))  # Resize to match image

        # Convert image tensor to NumPy format for visualization
        image = single_image.squeeze().detach().cpu().numpy()

        if image.ndim == 3 and image.shape[0] == 1:  # Grayscale image with shape (1, H, W)
            image = np.squeeze(image)  # Shape becomes (H, W)
            image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3) for RGB visualization
        elif image.ndim == 3 and image.shape[0] == 3:  # RGB image with shape (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)
        elif image.ndim == 2:  # Already a grayscale image (H, W)
            image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)

        # Rescale the image to the range [0, 255]
        image = (image * 255).astype(np.uint8)

        # Prepare the heatmap
        heatmap = np.uint8(255 * heatmap)  # Scale heatmap to [0, 255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Blend the heatmap with the original image
        blended_image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

        # Plot the result in the appropriate subplot
        ax = axes[i]
        ax.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        ax.axis('off')  # Hide axes
        ax.set_title(f"Prediction: {class_names[class_index]}")  # Set title with prediction

# Show the plot with multiple images in rows and columns
plt.tight_layout()
plt.show()
