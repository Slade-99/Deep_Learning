import os

# Define the folder path
folder = '/home/azwad/Downloads/Completed'

# Keywords and corresponding counts
keywords = {
    '_pneumonia': 0,
    '_normal': 0,
    '_abnormal': 0,
    '_effusion': 0
}

# Initialize a counter and list for images not matching any class
not_in_classes = 0
not_in_classes_files = []

# Get the list of files from the folder
files = os.listdir(folder)

# Count how many files contain each keyword
for file in files:
    file_lower = file.lower()  # Make case insensitive
    matched = False
    for keyword in keywords:
        if keyword in file_lower:
            keywords[keyword] += 1
            matched = True
            break
    if not matched:
        not_in_classes += 1
        not_in_classes_files.append(file)  # Store file name if it doesn't match any class

# Print the total count of images in the folder
total_images = len(files)
print(f"Total images: {total_images}")

# Print the counts for each class
for keyword, count in keywords.items():
    print(f"{keyword.capitalize()}: {count} images")

# Print the number of images not in any class
if not_in_classes > 0:
    print(f"Not in any class: {not_in_classes} images")
    print("Images not in any class:")
    for file in not_in_classes_files:
        print(file)  # Print the names of images that don't match any class
else:
    print("All images are categorized.")
