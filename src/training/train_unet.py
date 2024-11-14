import sys
import os
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from preprocessing.data_loader import load_dataset
from models.unet import unet_model


# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract model and training parameters from config
IMG_HEIGHT = config['model']['image_height']
IMG_WIDTH = config['model']['image_width']
BATCH_SIZE = config['model']['batch_size']
EPOCHS = config['model']['epochs']
MODEL_SAVE_PATH = os.path.join(config['paths']['models'], 'unet_model.keras')

# Load dataset
print("Loading dataset...")
images, masks = load_dataset()
print("Dataset loaded.")
print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")

# Define U-Net model
print("Building U-Net model...")
model = unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model built and compiled.")

# Setup callbacks for saving the model and early stopping
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
print("Starting training...")
history = model.fit(
    images, masks,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,  # Reserve 20% of data for validation
    callbacks=[checkpoint, early_stopping]
)

# Print summary of training history
print("Training completed.")
print(f"Model saved to: {MODEL_SAVE_PATH}")
