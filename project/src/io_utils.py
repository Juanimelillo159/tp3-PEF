"""
Funciones de E/S y utilidades para el procesamiento de imágenes.

Este módulo proporciona un conjunto de funciones de utilidad para manejar archivos de imagen,
incluyendo la carga, guardado y hashing de imágenes, así como la conversión
entre diferentes formatos de imagen.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    Carga una imagen desde una ruta de archivo.

    Args:
        image_path (str): La ruta al archivo de imagen.

    Returns:
        numpy.ndarray: La imagen cargada.
    """
    return cv2.imread(str(image_path))


def save_image(image, save_path):
    """
    Guarda una imagen en una ruta de archivo.

    Args:
        image (numpy.ndarray): La imagen a guardar.
        save_path (str): La ruta donde guardar la imagen.
    """
    cv2.imwrite(str(save_path), image)


def get_image_hash(image):
    """
    Calcula el hash SHA256 de una imagen.

    Args:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        str: El hash SHA256 de la imagen.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()


def pil_to_cv2(pil_image):
    """
    Convierte una imagen PIL a una imagen de OpenCV.

    Args:
        pil_image (PIL.Image.Image): La imagen PIL de entrada.

    Returns:
        numpy.ndarray: La imagen de OpenCV convertida.
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """
    Convierte una imagen de OpenCV a una imagen PIL.

    Args:
        cv2_image (numpy.ndarray): La imagen de OpenCV de entrada.

    Returns:
        PIL.Image.Image: La imagen PIL convertida.
    """
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
