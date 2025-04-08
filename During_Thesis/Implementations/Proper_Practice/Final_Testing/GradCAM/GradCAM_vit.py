import torch
import torch.nn as nn
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.MobileViT_S.model_gradcam import model
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.LeViT.model_new import model
#from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.MobileNet_V2.model_new import model
from During_Thesis.Implementations.Proper_Practice.Final_Testing.Model.Swin.model_new import model
from torchsummary import summary
from During_Thesis.Implementations.Preprocessings.Private_Dataset_Preprocessings.Testing_prepare_private_dataset import test_dataloader ,eval_transforms
import numpy as np
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['normal', 'pneumonia', 'abnormal' ]

# Preparing the Trained Model
model_path = '/home/azwad/Works/Model_Weights/LeViT.pth.tar'
model = model.to(device)
learning_rate = 0.0001
#checkpoint = torch.load(model_path)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#model.load_state_dict(checkpoint['state_dict'])
model.eval()

img = None
train_dataloader = test_dataloader

# Prepare the subplot grid
num_images = 20  # Number of images to display
rows = 4  # Number of rows
cols = 5  # Number of columns
fig, axes = plt.subplots(rows, cols, figsize=(10,10))  # Adjust the size accordingly
axes = axes.flatten()  # Flatten axes array for easier indexing

for idx, (images, labels) in enumerate(train_dataloader):
    if idx >= 1:
        break  # We only need to process the first batch

    for i in range(min(num_images, images.shape[0])):  # Display up to num_images images from the batch
        single_image = images[i].unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Perform prediction
        pred = model(single_image)
        class_index = pred.argmax(dim=1).item()

        # Compute gradients with respect to the predicted class
        
        pred[:, class_index].backward(retain_graph=True)

        # Retrieve gradients and activations
        gradients = model.get_activations_gradient()
        print(gradients)
        pooled_gradients = torch.mean(gradients, dim=2)
        #print(pooled_gradients)
        activations = model.get_activations(single_image).detach()
        

        # Weight the activations by the pooled gradients
        #for j in range(activations.shape[1]):  # Dynamically handle channel count
        #    activations[:, j, :, :] *= pooled_gradients[j]
        weights = pooled_gradients.unsqueeze(-1)  # Shape: [1, 16, 1]
        weighted_activations = activations * weights

        # Compute the heatmap
        heatmap = torch.mean(weighted_activations, dim=2).squeeze()
        heatmap = heatmap / torch.max(heatmap)  # Normalize
        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0.1)  
        #heatmap = heatmap / np.max(heatmap + 1e-8)
        heatmap = heatmap.reshape(4, 4)
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
        blended_image = cv2.addWeighted(image, 0.6, heatmap, 0.3, 0)

        # Plot the result in the appropriate subplot
        ax = axes[i]
        ax.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        ax.axis('off')  # Hide axes
        ax.set_title(f"Prediction: {class_names[class_index]}")  # Set title with prediction

# Show the plot with multiple images in rows and columns
plt.tight_layout()
#plt.show()
