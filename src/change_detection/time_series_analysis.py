import numpy as np

def detect_changes(image_sequence):
    changes = []
    for i in range(1, len(image_sequence)):
        diff = np.abs(image_sequence[i] - image_sequence[i-1])
        changes.append(diff)
    return changes
