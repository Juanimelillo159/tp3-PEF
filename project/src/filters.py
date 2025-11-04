import cv2
import numpy as np
from .cache import memoize

@memoize
def apply_sobel(image):
    """Applies the Sobel filter to an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    sobel = np.uint8(sobel / sobel.max() * 255)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

@memoize
def apply_canny(image):
    """Applies the Canny edge detector to an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

@memoize
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies a Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

@memoize
def apply_sharpen(image):
    """Applies a sharpening filter to an image."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def apply_random_hue_shift(image):
    """Applies a random hue shift to an image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = np.random.randint(0, 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
