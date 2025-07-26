import os
import json

def generate_dataset_json(directory):
    # Initialize the dataset dictionary
    dataset = {"labels": []}

    # Loop through all the subdirectories in the given directory
    for class_id, class_folder in enumerate(os.listdir(directory)):
        class_folder_path = os.path.join(directory, class_folder)

        # Only process directories (subfolders)
        if os.path.isdir(class_folder_path):
            # Loop through all the images in the subdirectory
            for image_name in os.listdir(class_folder_path):
                # Ensure it's an image file (you can expand this to other file formats if needed)
                if image_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_folder, image_name)
                    dataset["labels"].append([image_path, class_id])

    # Save the generated dataset to a JSON file
    with open("dataset.json", "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    print("dataset.json file has been generated successfully!")

# Example: You can provide any directory path here
input_directory = input("Enter the path to the dataset directory: ")
generate_dataset_json(input_directory)
