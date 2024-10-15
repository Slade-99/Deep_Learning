import os
import imageio
from PIL import Image

# Define the source and destination folder paths
source_folder = '/home/azwad/Downloads/HEIC_FILES'  # Folder with .heic files
destination_folder = '/home/azwad/Downloads/Converted'  # Folder to save converted .jpg files

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get the list of HEIC files from the source folder
files = [f for f in os.listdir(source_folder) if f.lower().endswith('.heic')]

# Process each .heic file and convert to .jpg
for file in files:
    heic_file_path = os.path.join(source_folder, file)
    
    try:
        # Read the .heic file using imageio
        heif_image = imageio.v3.imread(heic_file_path)

        # Convert the image to a PIL Image
        image = Image.fromarray(heif_image)
        
        # Save the image as .jpg in the destination folder
        jpg_file_path = os.path.join(destination_folder, file.replace('.heic', '.jpg'))
        image.save(jpg_file_path, "JPEG")
        print(f"Converted and saved: {file} -> {jpg_file_path}")
    
    except Exception as e:
        print(f"Skipping {file}: {e}")
