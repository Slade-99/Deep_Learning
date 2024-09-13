import os
import csv

# Define the folder containing the CXR images
folder_path = '/home/azwad/Datasets/Thesis DataSet-20240825T121325Z-001/Cropped'

# Define the mapping from image name to label
label_mapping = {
    'cough': 2,
    'pneumonia': 1,
    'normal': 0
}

# List to store the filename and corresponding label
data = []

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a CXR image (assuming images have a .jpg or .png extension)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Extract the label part from the filename
        for key in label_mapping.keys():
            if key in filename:
                # Append the filename and its corresponding label to the data list
                data.append([filename, label_mapping[key]])
                break

# Define the path for the CSV file
csv_file_path = '/home/azwad/Datasets/Thesis DataSet-20240825T121325Z-001/output.csv'

# Write the data to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Label'])  # Write header
    writer.writerows(data)  # Write the rows of data

print(f"CSV file has been created: {csv_file_path}")
