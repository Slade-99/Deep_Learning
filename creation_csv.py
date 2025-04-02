



import os
import pandas as pd

def extract_details_from_filenames(folder_path, output_csv="image_data.csv"):
    data = []
    
    # Iterate through all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Check if the file is a .jpg image
            try:
                # Remove the file extension and split the filename by '_'
                parts = filename.replace(".jpg", "").split("_")
                
                if len(parts) == 4:  # Ensure the filename has the expected format
                    serial, gender, age, class_name = parts
                    data.append([serial, gender, age, class_name, filename])
                else:
                    print(f"Skipping invalid filename format: {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Serial", "Gender", "Age", "Class", "Filename"])
    
    # Save to CSV
    output_path = os.path.join(folder_path, output_csv)
    df.to_csv(output_path, index=False)
    print(f"CSV file saved: {output_path}")

# Example Usage:
folder_path = "/mnt/hdd/dataset_collections/Private/raw/unified"  # Change this to your actual folder path
extract_details_from_filenames(folder_path)
