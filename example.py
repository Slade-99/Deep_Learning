from PIL import Image
import matplotlib.pyplot as plt

# Load the two images
image1 = Image.open("/home/azwad/Downloads/InvoSparseNet_Paper/Corrected/cm_nih.png")  # Replace with your actual file names
image2 = Image.open("/home/azwad/Downloads/InvoSparseNet_Paper/Corrected/cm_primary.png")

# Ensure both images have the same height
if image1.height != image2.height:
    new_height = min(image1.height, image2.height)
    image1 = image1.resize((int(image1.width * new_height / image1.height), new_height))
    image2 = image2.resize((int(image2.width * new_height / image2.height), new_height))

# Create a new blank image with the combined width
new_width = image1.width + image2.width
combined_image = Image.new("RGB", (new_width, image1.height))

# Paste the images side by side
combined_image.paste(image1, (0, 0))
combined_image.paste(image2, (image1.width-200, 0))

# Show the final combined image
plt.imshow(combined_image)
plt.axis("off")
plt.show()

# Optionally, save the output image
combined_image.save("combined_image.png")
