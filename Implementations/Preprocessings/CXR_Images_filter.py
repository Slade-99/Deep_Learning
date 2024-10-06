import os
from PIL import Image

# Define the source folder containing the original images
source_folder = '/home/azwad/Datasets/Benchmark_Dataset/Unified'

# Define the destination folder for transformed images
destination_folder = '/home/azwad/Datasets/Benchmark_Dataset/Filtered/Images'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Define the target size
target_size = (256, 256)

# Process each image in the source folder
for filename in os.listdir(source_folder):
    # Construct full file path
    file_path = os.path.join(source_folder, filename)
    
    # Check if the file is an image (you can add more extensions if needed)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Open the image
        with Image.open(file_path) as img:
            # Resize the image
            img_resized = img.resize(target_size)
            
            # Convert the image to grayscale
            img_gray = img_resized.convert('L')
            
            # Construct the full path for the new file
            new_file_path = os.path.join(destination_folder, filename)
            
            # Save the transformed image to the destination folder
            img_gray.save(new_file_path)

print(f"All images have been processed and saved to {destination_folder}")
