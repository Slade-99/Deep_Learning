import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the image
image_path = '/home/azwad/Works/Deep_Learning/Usable_Datasets/Private/Cropped/12_male_60_pneumonia.jpg'
image = Image.open(image_path)

# Convert the image to grayscale using transforms
transform_to_grayscale = transforms.Grayscale(num_output_channels=1)
grayscale_image = transform_to_grayscale(image)

# Convert to a NumPy array for OpenCV processing
image_np = np.array(grayscale_image)

# Apply CLAHE using OpenCV
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image_np)

# Apply Median Blur using OpenCV
blurred_image = cv2.medianBlur(clahe_image, 5)  # Kernel size of 5

# Convert the processed image back to PIL Image format for further processing in PyTorch
final_image = Image.fromarray(blurred_image)

# Convert the processed image to tensor
transform_to_tensor = transforms.ToTensor()
image_tensor = transform_to_tensor(final_image)

# Plot the original and processed images side-by-side
plt.figure(figsize=(12, 6))

# Plot original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Plot processed image (CLAHE + Median Blur)
plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title("Processed Image (CLAHE + Median Blur)")
plt.axis('off')

# Show the plots
plt.show()

# Print the shape of the processed image tensor
print("Tensor shape:", image_tensor.shape)
