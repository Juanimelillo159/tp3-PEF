import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    """Loads an image from a file path."""
    return cv2.imread(str(image_path))

def save_image(image, save_path):
    """Saves an image to a file path."""
    cv2.imwrite(str(save_path), image)

def get_image_hash(image):
    """Calculates the SHA256 hash of an image."""
    return hashlib.sha256(image.tobytes()).hexdigest()

def pil_to_cv2(pil_image):
    """Converts a PIL image to an OpenCV image."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Converts an OpenCV image to a PIL image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
