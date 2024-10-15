import os
import shutil

# Define the paths to both folders
folder1 = '/home/azwad/Downloads/new_challenge/corrected'  # Folder with names like (number% %something)
folder2 = '/home/azwad/Downloads/final_middel_part'  # Folder with names like (number_something_something.jpg)

# Get the list of files from both folders
files_folder1 = os.listdir(folder1)
files_folder2 = os.listdir(folder2)

# Process each file in folder1
for file1 in files_folder1:
    if '%' in file1:
        # Extract the number from the first folder's file name
        number = file1.split('%')[0].strip()
        
        # Look for the corresponding file in folder2
        for file2 in files_folder2:
            if file2.startswith(number + '_'):
                # Extract the middle part from the second folder's file name
                middle_part = file2[len(number):].split('.jpg')[0]
                
                # Create the new file name for the first folder without replacing the part after the second '%'
                parts = file1.split('%')
                new_name = f"{parts[0]}%{middle_part}%{parts[2]}"
                
                # Define the full old and new paths
                old_path = os.path.join(folder1, file1)
                new_path = os.path.join(folder1, new_name)
                
                # Rename the file
                shutil.move(old_path, new_path)
                print(f"Renamed: {file1} -> {new_name}")
                break
