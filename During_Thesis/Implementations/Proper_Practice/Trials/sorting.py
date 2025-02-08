import os
import shutil
import pandas as pd

# Path to the folder containing the images and the CSV file
images_folder = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/images-224'  # Replace with your images folder path
csv_file_path = '/home/azwad/Works/PadChest/PadChest_dataset_full/modified2.csv'  # Replace with your CSV file path
output_folder = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/Classified/train'  # Folder where the sorted images will be placed

# Number of samples to keep per label
max_samples_per_label = 3300

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Dictionary to keep track of how many images per label we have
label_counts = {}

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over each row in the CSV to organize the images
for _, row in df.iterrows():
    image_filename = row[0]
    label = row[1]
    
    # Check if image file exists in the images folder
    image_path = os.path.join(images_folder, image_filename)
    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found in {images_folder}. Skipping.")
        continue
    
    # Create label folder if it doesn't exist
    label_folder = os.path.join(output_folder, str(label))
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    
    # Initialize label count if it doesn't exist
    if label not in label_counts:
        label_counts[label] = 0
    
    # Check if we have less than the max allowed samples for this label
    if label_counts[label] < max_samples_per_label:
        # Move the image to the label folder
        destination_path = os.path.join(label_folder, image_filename)
        shutil.copy(image_path, destination_path)
        label_counts[label] += 1

# Print the number of images copied per label
for label, count in label_counts.items():
    print(f"Label '{label}' has {count} images.")

print("Image sorting completed!")
