import os
from PIL import Image

def check_image_resolution_and_mode(image_path):
    try:
        # Open the image using PIL
        with Image.open(image_path) as img:
            # Check the resolution (width and height must be non-zero)
            print(img.width, img.height)
            if img.width == 0 or img.height == 0:
                return f"Invalid resolution: {img.size} (width x height) for {image_path}"

            # Check if the image is grayscale or RGB
            if img.mode == 'L':  # 'L' mode means grayscale
                return f"Grayscale image: {image_path}"
            elif img.mode == 'RGB':  # 'RGB' mode means color image
                return f"RGB image: {image_path}"
            else:
                return f"Image mode {img.mode} not supported for {image_path}"
    except Exception as e:
        return f"Error processing {image_path}: {e}"

def check_images_in_directory(directory_path):
    invalid_images = []

    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file is an image (by common extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, file)
                result = check_image_resolution_and_mode(image_path)
                if result:
                    invalid_images.append(result)

    return invalid_images

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    
    # Check all images in the directory and subdirectories
    results = check_images_in_directory(directory)

    if results:
        print("\nInvalid images found:")
    else:
        print("All images are valid!")
