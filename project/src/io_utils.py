"""
I/O and utility functions for image processing.

This module provides a set of utility functions for handling image files,
including loading, saving, and hashing images, as well as converting
between different image formats.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    Loads an image from a file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image.
    """
    return cv2.imread(str(image_path))


def save_image(image, save_path):
    """
    Saves an image to a file path.

    Args:
        image (numpy.ndarray): The image to be saved.
        save_path (str): The path to save the image to.
    """
    cv2.imwrite(str(save_path), image)


def get_image_hash(image):
    """
    Calculates the SHA256 hash of an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        str: The SHA256 hash of the image.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()


def pil_to_cv2(pil_image):
    """
    Converts a PIL image to an OpenCV image.

    Args:
        pil_image (PIL.Image.Image): The input PIL image.

    Returns:
        numpy.ndarray: The converted OpenCV image.
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """
    Converts an OpenCV image to a PIL image.

    Args:
        cv2_image (numpy.ndarray): The input OpenCV image.

    Returns:
        PIL.Image.Image: The converted PIL image.
    """
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
