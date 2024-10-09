import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

# Function to enhance image
def enhance_image(image_path, output_path):
    # Open image using OpenCV
    img = cv2.imread(image_path)
    
    # Resize image if needed
    img = cv2.resize(img, (720, 720))
    
    # Step 1: Sharpening the image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Step 2: Convert to PIL image for further enhancement
    pil_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    # Step 3: Adjust contrast using Pillow
    enhancer = ImageEnhance.Contrast(pil_img)
    contrast_img = enhancer.enhance(1.5)  # You can tweak the factor here
    
    # Step 4 (Optional): Denoise using OpenCV
    denoised_img = cv2.fastNlMeansDenoisingColored(np.array(contrast_img), None, 10, 10, 7, 21)
    
    # Save the enhanced image
    final_img = Image.fromarray(denoised_img)
    final_img.save(output_path)

# Folder paths
input_folder = "/home/azwad/Datasets/Thesis DataSet/To be edited"
output_folder = "/home/azwad/Datasets/Thesis DataSet/Reformed"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image
for img_file in os.listdir(input_folder):
    if img_file.endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)
        enhance_image(input_path, output_path)

print("Image enhancement completed for 40 images.")
