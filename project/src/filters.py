"""
Una colección de filtros de procesamiento de imágenes.

Este módulo proporciona un conjunto de funciones para aplicar diversos filtros de
procesamiento de imágenes a las imágenes, incluida la detección de bordes, el desenfoque,
la nitidez y la manipulación del color. Las funciones están diseñadas para ser utilizadas
como parte del pipeline de procesamiento de imágenes.
"""
import cv2
import numpy as np
from .cache import memoize


@memoize
def apply_sobel(image):
    """
    Aplica el filtro Sobel a una imagen para detectar bordes.

    Args:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen con el filtro Sobel aplicado.
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
    Aplica el detector de bordes Canny a una imagen.

    Args:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen con el detector de bordes Canny aplicado.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)


@memoize
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Aplica un desenfoque Gaussiano a una imagen.

    Args:
        image (numpy.ndarray): La imagen de entrada.
        kernel_size (tuple, optional): El tamaño del kernel. Por defecto es (5, 5).

    Returns:
        numpy.ndarray: La imagen desenfocada.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


@memoize
def apply_sharpen(image):
    """
    Aplica un filtro de nitidez a una imagen.

    Args:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen enfocada.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def apply_random_hue_shift(image):
    """
    Aplica un cambio de tono aleatorio a una imagen.

    Esta función no está memoizada porque no es determinista.

    Args:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen con un cambio de tono aleatorio aplicado.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = np.random.randint(0, 180)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
