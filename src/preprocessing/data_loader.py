import os
from PIL import Image

def load_image(image_path):
    return Image.open(image_path)

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = load_image(os.path.join(folder_path, filename))
            images.append(img)
    return images
