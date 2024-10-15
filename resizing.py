import os
from PIL import Image

# Define the source and destination folder paths
source_folder = '/home/azwad/Downloads/Completed'  # Folder with the original images
destination_folder = '/home/azwad/Downloads/Resized'  # Folder to save resized images

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get the list of image files from the source folder
files = os.listdir(source_folder)

# Process each file and resize to 1024x1024
for file in files:
    file_path = os.path.join(source_folder, file)
    
    # Open an image file
    with Image.open(file_path) as img:
        # Resize the image to 1024x1024
        img_resized = img.resize((1024, 1024))
        
        # Save the resized image to the destination folder
        img_resized.save(os.path.join(destination_folder, file))
        print(f"Resized and saved: {file}")
