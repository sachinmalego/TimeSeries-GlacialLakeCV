import cv2
import numpy as np

def apply_morphology(segmentation_mask):
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask
