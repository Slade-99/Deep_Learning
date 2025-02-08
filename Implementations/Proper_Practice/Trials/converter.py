from PIL import Image
import os
import numpy as np

def convert_to_8bit(input_folder, output_folder):
    # Check if the output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        # Only process PNG files
        if file_name.endswith('.png'):
            # Full path to the input image
            input_path = os.path.join(input_folder, file_name)
            
            try:
                # Open the image
                with Image.open(input_path) as img:
                    # Print the image mode for debugging
                    print(f"Processing {file_name} with mode: {img.mode}")
                    
                    if img.mode == 'I;16':
                        # If the image is in I;16 (16-bit grayscale), scale the pixel values to 0-255 range
                        img = np.array(img)  # Convert to numpy array for manipulation
                        img = (img / 256).astype(np.uint8)  # Normalize from 16-bit to 8-bit by dividing by 256
                        img = Image.fromarray(img)  # Convert back to PIL image
                        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)  # Convert to 8-bit with a palette
                    elif img.mode == 'I':
                        # For 'I' mode (16-bit signed integer), convert directly to 'L' (grayscale) first
                        img = img.convert('L')
                        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)  # Convert to 8-bit with a palette
                    elif img.mode in ('RGB', 'RGBA'):
                        # RGB/RGBA to 8-bit using adaptive palette
                        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                    elif img.mode == 'L':
                        # Grayscale (L) to 8-bit using adaptive palette
                        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
                    else:
                        # If the mode is unsupported, skip this image
                        print(f"Skipping {file_name} due to unsupported mode ({img.mode})")
                        continue

                    # Define output file path
                    output_path = os.path.join(output_folder, file_name)
                    
                    # Save the converted image to the output folder
                    img.save(output_path)
                    print(f"Converted and saved: {file_name}")
            
            except Exception as e:
                # If there is an error (e.g., corrupted file), print the error
                print(f"Error processing {file_name}: {e}")

# Set your input and output folder paths here



input_folder = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/Classified/train/pneumonia'
output_folder = '/home/azwad/Works/PadChest/PadChest_dataset_full/images-224/Fresh/train/pneumonia'

convert_to_8bit(input_folder, output_folder)


