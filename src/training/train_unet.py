import tensorflow as tf
from src.models.unet import build_unet
from src.preprocessing.data_loader import load_images_from_folder

def train_unet(data_path, epochs=10):
    model = build_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    train_images = load_images_from_folder(data_path)
    # Prepare dataset, labels, and run training here
    # model.fit(train_images, labels, epochs=epochs)
    print("Training complete")

if __name__ == "__main__":
    train_unet("data/processed/")
