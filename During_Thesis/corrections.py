import cv2
import os

# Define font parameters
font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a readable font
font_scale = 2  # Increase the size
font_thickness = 3  # Make it bold
color = (0, 0, 0)  # Black text (change if needed)

# Directory containing figures
input_folder = "/home/azwad/Downloads/InvoSparseNet_Paper/Original"  # Change to your folder path
output_folder = "/home/azwad/Downloads/InvoSparseNet_Paper/Corrected"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(input_folder, filename))

        # Example: Adding a title text
        cv2.putText(img, "Figure Title", (50, 100), font, font_scale, color, font_thickness)

        # Save the updated image
        cv2.imwrite(os.path.join(output_folder, filename), img)

print("Updated images saved in", output_folder)
