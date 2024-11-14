import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract paths and model parameters from config
IMG_HEIGHT = config['model']['image_height']
IMG_WIDTH = config['model']['image_width']
DATA_IMAGES_PATH = config['paths']['data_images']
DATA_MASKS_PATH = config['paths']['data_masks']

def load_images_from_folder(folder_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """Load images from a specified folder, resize them, and normalize."""
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)
    return np.array(images)

def load_dataset():
    """
    Load images and masks using paths from the config file.
    
    Returns:
    - (np.array, np.array): Tuple of numpy arrays containing images and corresponding masks.
    """
    # Load images and masks
    images = load_images_from_folder(DATA_IMAGES_PATH, IMG_HEIGHT, IMG_WIDTH)
    masks = load_images_from_folder(DATA_MASKS_PATH, IMG_HEIGHT, IMG_WIDTH)

    # Ensure masks are grayscale (1 channel)
    masks = np.expand_dims(masks[:, :, :, 0], axis=3)  # Convert to (height, width, 1)

    return images, masks

if __name__ == "__main__":
    images, masks = load_dataset()
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)