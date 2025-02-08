import os
import pandas as pd

# Path to the folder containing the images and the CSV file
images_folder = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/Classified/train/pneumonia'  # Replace with your images folder path
csv_file_path = '/home/azwad/Works/PadChest/PadChest_dataset_full/modified.csv'  # Replace with your CSV file path

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Loop over each row in the CSV
for _, row in df.iterrows():
    image_filename = row[0]  # Image name from the first column
    label = row[1]  # Label from the second column
    
    # Check if the label is "L" and if the image file exists in the images folder
    if label == "L":
        image_path = os.path.join(images_folder, image_filename)
        
        # If the image exists, delete it
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted {image_filename}")
        else:
            print(f"Image {image_filename} not found in {images_folder}. Skipping.")

print("Image deletion completed!")
