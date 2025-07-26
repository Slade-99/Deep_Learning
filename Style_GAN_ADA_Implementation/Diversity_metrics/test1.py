import torch
from lpips import LPIPS
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def load_images(image_folder, image_size=(256, 256)):
    """Loads images from a folder and applies transformations."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    images = []
    for file in os.listdir(image_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(image_folder, file)).convert('RGB')
            images.append(transform(img))
    return torch.stack(images)

def compute_lpips_diversity(images, lpips_model):
    """Computes pairwise LPIPS distances and returns the mean as a diversity metric."""
    num_images = images.shape[0]
    distances = []
    
    for i in range(num_images):
        for j in range(i + 1, num_images):
            dist = lpips_model(images[i].unsqueeze(0), images[j].unsqueeze(0))
            distances.append(dist.item())
    
    return np.mean(distances) if distances else 0.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_folder = "/mnt/hdd/Dataset_augmentation/generated_images/unified/"  # Change this to your image folder
    images = load_images(image_folder)
    images = images.to(device)
    lpips_model = LPIPS(net='vgg').to(device)  # Choose 'alex', 'vgg', or 'squeeze'
    
    diversity_score = compute_lpips_diversity(images, lpips_model)
    print(f"LPIPS Diversity Score: {diversity_score}")
