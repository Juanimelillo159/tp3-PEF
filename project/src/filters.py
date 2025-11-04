"""
A collection of image processing filters.

This module provides a set of functions for applying various image processing
filters to images, including edge detection, blurring, sharpening, and color
manipulation. The functions are designed to be used as part of the image
processing pipeline.
"""
import cv2
import numpy as np
from .cache import memoize


@memoize
def apply_sobel(image):
    """
    Applies the Sobel filter to an image to detect edges.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with the Sobel filter applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    sobel = np.uint8(sobel / sobel.max() * 255)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)


@memoize
def apply_canny(image):
    """
    Applies the Canny edge detector to an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with the Canny edge detector applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)


@memoize
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Applies a Gaussian blur to an image.

    Args:
        image (numpy.ndarray): The input image.
        kernel_size (tuple, optional): The size of the kernel. Defaults to (5, 5).

    Returns:
        numpy.ndarray: The blurred image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


@memoize
def apply_sharpen(image):
    """
    Applies a sharpening filter to an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The sharpened image.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def apply_random_hue_shift(image):
    """
    Applies a random hue shift to an image.

    This function is not memoized because it is non-deterministic.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with a random hue shift applied.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = np.random.randint(0, 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
