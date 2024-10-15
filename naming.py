import os
import shutil

# Define the paths to your folders
old_labellings_folder = '/home/azwad/Downloads/New_Labellings'
new_labellings_folder = '/home/azwad/Downloads/Old_Labellings'

# Get lists of files in both folders
old_files = os.listdir(old_labellings_folder)
new_files = os.listdir(new_labellings_folder)

# Create a mapping of old file names based on their numeric prefix
old_name_map = {}
for old_file in old_files:
    # Extract the numeric prefix
    prefix = old_file.split('_')[0]
    old_name_map[prefix] = old_file

# Rename the new files based on the old names
for new_file in new_files:
    # Extract the numeric prefix
    prefix = new_file.split('_')[0]
    if prefix in old_name_map:
        old_name = old_name_map[prefix]
        new_name = old_name  # Use the old name for renaming
        # Define full path for new and old files
        old_file_path = os.path.join(old_labellings_folder, old_name)
        new_file_path = os.path.join(new_labellings_folder, new_file)

        # Rename the new file to the old file name
        new_file_new_name = os.path.join(new_labellings_folder, new_name)
        shutil.move(new_file_path, new_file_new_name)

print("Renaming completed.")
