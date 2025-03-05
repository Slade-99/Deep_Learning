import os
import shutil
import random

def split_images_into_train_val_test(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Splits images from class folders into train, val, and test folders.

    Args:
        source_dir (str): Path to the source directory (NIH_Filtered).
        target_dir (str): Path to the target directory (NIH).
        train_ratio (float): Ratio of data for training.
        val_ratio (float): Ratio of data for validation.
        test_ratio (float): Ratio of data for testing.
        random_seed (int): Random seed for reproducibility.
    """

    random.seed(random_seed)

    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    for class_folder in class_folders:
        source_class_dir = os.path.join(source_dir, class_folder)
        train_class_dir = os.path.join(train_dir, class_folder)
        val_class_dir = os.path.join(val_dir, class_folder)
        test_class_dir = os.path.join(test_dir, class_folder)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        image_files = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] #add more extensions if needed.
        random.shuffle(image_files)

        train_split = int(train_ratio * len(image_files))
        val_split = int((train_ratio + val_ratio) * len(image_files))

        train_images = image_files[:train_split]
        val_images = image_files[train_split:val_split]
        test_images = image_files[val_split:]

        for image in train_images:
            source_path = os.path.join(source_class_dir, image)
            destination_path = os.path.join(train_class_dir, image)
            shutil.copy2(source_path, destination_path)

        for image in val_images:
            source_path = os.path.join(source_class_dir, image)
            destination_path = os.path.join(val_class_dir, image)
            shutil.copy2(source_path, destination_path)

        for image in test_images:
            source_path = os.path.join(source_class_dir, image)
            destination_path = os.path.join(test_class_dir, image)
            shutil.copy2(source_path, destination_path)

    print("Image split complete.")

# Example Usage:
source_folder = '/mnt/hdd/dataset_collections/NIH_Filtered'  # Replace with your NIH_Filtered folder path
target_folder = '/mnt/hdd/dataset_collections/NIH_Filtered_2'  # Replace with your NIH folder path

split_images_into_train_val_test(source_folder, target_folder)