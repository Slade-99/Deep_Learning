
import os
import matplotlib.pyplot as plt
from PIL import Image
import math

# Path to the folder that contains the model folders
base_folder = "/home/azwad/Downloads/Dataset_Paper/Model_Specific"

# Get sorted list of model folders
model_names = sorted(os.listdir(base_folder))

# File names
cm_name = "cm.png"
roc_name = "roc.png"
gradcam_names = ["g_pneumonia.png", "g_abnormal.png", "g_normal.png"]

def load_image(path):
    return Image.open(path)

# ========== Helper for 2-row layout with center alignment ==========
def plot_images_centered(images, titles, save_path, title_text):
    total = len(images)
    top_row = math.ceil(total / 2)
    bottom_row = total - top_row
    max_cols = max(top_row, bottom_row)
    fig, axs = plt.subplots(2, max_cols, figsize=(5 * max_cols, 10))

    # Top row
    for i in range(max_cols):
        ax = axs[0, i]
        if i < top_row:
            ax.imshow(images[i])
            ax.set_title(titles[i])
        else:
            ax.axis('off')
        ax.axis('off')

    # Bottom row
    pad = (max_cols - bottom_row) // 2  # how many blanks to pad left
    for i in range(max_cols):
        ax = axs[1, i]
        if pad <= i < pad + bottom_row:
            img_idx = top_row + (i - pad)
            ax.imshow(images[img_idx])
            ax.set_title(titles[img_idx])
        else:
            ax.axis('off')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

# ========== 1. Confusion Matrices ==========
cm_imgs, cm_titles = [], []
for model in model_names:
    path = os.path.join(base_folder, model, cm_name)
    if os.path.exists(path):
        cm_imgs.append(load_image(path))
        cm_titles.append(model)

plot_images_centered(cm_imgs, cm_titles, "all_confusion_matrices.png", "Confusion Matrices")

# ========== 2. ROC Curves ==========
roc_imgs, roc_titles = [], []
for model in model_names:
    path = os.path.join(base_folder, model, roc_name)
    if os.path.exists(path):
        roc_imgs.append(load_image(path))
        roc_titles.append(model)

plot_images_centered(roc_imgs, roc_titles, "all_roc_curves.png", "ROC Curves")

# ========== 3. GradCAMs ==========
gradcam_matrix = []
for model in model_names:
    row_imgs = []
    for fname in gradcam_names:
        path = os.path.join(base_folder, model, fname)
        if os.path.exists(path):
            row_imgs.append(load_image(path))
    gradcam_matrix.append((model, row_imgs))

rows, cols = len(gradcam_matrix), len(gradcam_names)

# Create a figure with a size appropriate to the number of rows and columns
gradcam_matrix = []
for model in model_names:
    row_imgs = []
    for fname in gradcam_names:
        path = os.path.join(base_folder, model, fname)
        if os.path.exists(path):
            row_imgs.append(load_image(path))
    gradcam_matrix.append((model, row_imgs))

rows, cols = len(gradcam_matrix), len(gradcam_names)

# Create a figure with a size appropriate to the number of rows and columns
plt.figure(figsize=(3 * cols, 3 * rows))

# Loop through the rows (models)
for row_idx, (model, images) in enumerate(gradcam_matrix):
    for col_idx, img in enumerate(images):
        index = row_idx * cols + col_idx + 1
        plt.subplot(rows, cols, index)
        plt.imshow(img)
        plt.axis('off')

        # Set column titles (class names) at the top of each column
        if row_idx == 0:  # Only add column titles once (for the first row)
            class_name = gradcam_names[col_idx].split('_')[1].split('.')[0].capitalize()  # Extract class name
            plt.title(class_name)

        # Set row titles (model names) on the left side of each row using plt.text()
        if col_idx == 0:  # Only add row titles once (for the first column)
            plt.text(-0.1, 0.5, model, va='center', ha='right', fontsize=12, rotation=0, transform=plt.gca().transAxes)
plt.subplots_adjust(wspace=0.01, hspace=0.3)
# Adjust layout and show the plot
plt.tight_layout()
plt.show()
plt.close()
