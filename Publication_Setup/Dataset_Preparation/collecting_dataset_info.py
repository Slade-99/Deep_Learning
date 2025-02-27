### Checking Dataset Info ###
import random
import os
from PIL import Image
from pathlib import Path
dataset_directory = Path("/mnt/hdd/Datasets/")
dataset_path = dataset_directory / "Private_CXR"
train_dir = dataset_path / "train"
test_dir = dataset_path / "test"
val_dir = dataset_path / "val"

def walk_through_dir(dir_path):
    
    for dirpath , dirnames ,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath} ")


#walk_through_dir(dataset_path)


def display_random_image(dataset_path):
    img_path_list = list(dataset_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(img_path_list)
    image_class = random_image_path.parent.stem
    print(image_class)
    img = Image.open(random_image_path)
    
    ##Metadata
    print(f"Image_Class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    img.show()


display_random_image(dataset_path)
    