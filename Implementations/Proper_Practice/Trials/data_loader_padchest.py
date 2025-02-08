import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Function to load and preprocess a PNG image
def load_png_as_tensor(image_path):
    # Load the image using PIL
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    return image

# Custom Dataset class for PNG images
class PadChestPNGDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_png_as_tensor(image_path)  # Load and preprocess the PNG
        
        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),         # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Get all PNG file paths from a directory
def get_png_file_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):  # Handle PNG files
                image_paths.append(os.path.join(root, file))
    return image_paths

# Replace with the actual path to your PNG folder
png_folder_path = "/home/azwad/Works/Deep_Learning/Usable_Datasets/PadChest/Classified/train/abnormal"
image_paths = get_png_file_paths(png_folder_path)

# Debugging: Check the number of files found
print(f"Found {len(image_paths)} PNG files")
if len(image_paths) == 0:
    raise ValueError("No PNG files found. Check the folder path and file extensions.")

# Create the dataset
dataset = PadChestPNGDataset(image_paths, transform=transform)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Test the DataLoader
for batch in dataloader:
    print(batch.shape)  # Should print (batch_size, channels, height, width)




import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def visualize_image_from_dataloader(dataloader):
    """
    Visualize an image from the DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader object.
    """
    # Get a single batch from the DataLoader
    for images in dataloader:
        # Take the first image from the batch
        image = images[0]
        
        # Convert the tensor to a PIL image for visualization
        image_pil = TF.to_pil_image(image)
        
        # Plot the image
        plt.imshow(image_pil, cmap="gray")
        plt.axis("off")
        plt.show()
        
        break  # Exit after visualizing the first batch

# Use the function
visualize_image_from_dataloader(dataloader)