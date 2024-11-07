from src.preprocessing.data_loader import load_images_from_folder
from src.models.unet import build_unet
from src.postprocessing.morphological_ops import apply_morphology
from src.change_detection.time_series_analysis import detect_changes

def main_pipeline(data_path):
    # Load and preprocess data
    images = load_images_from_folder(data_path)
    # Initialize and train model
    model = build_unet()
    # Apply post-processing
    segmented_images = [apply_morphology(image) for image in images]
    # Detect changes in segmented areas over time
    changes = detect_changes(segmented_images)
    print("Pipeline complete")

if __name__ == "__main__":
    main_pipeline("data/processed/")
