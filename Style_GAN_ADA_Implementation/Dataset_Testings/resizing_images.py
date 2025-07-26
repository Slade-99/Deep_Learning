import os
from PIL import Image

def resize_image(image_path, output_path, new_width, new_height):
    try:
        with Image.open(image_path) as img:
            # Resize image while maintaining aspect ratio
            img_resized = img.resize((new_width, new_height))
            # Save the resized image to the same path, replacing the original
            img_resized.save(output_path)
            print(f"Image resized and saved: {output_path}")
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")

def resize_images_in_directory(directory_path, new_width, new_height):
    # Loop through all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, file)
                resize_image(image_path, image_path, new_width, new_height)

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    new_width = int(input("Enter the new width: "))
    new_height = int(input("Enter the new height: "))
    
    # Resize all images in the directory to the new resolution
    resize_images_in_directory(directory, new_width, new_height)

    print("Image resizing complete!")
