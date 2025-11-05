"""
Orquesta el pipeline de procesamiento de imágenes.

Este módulo es responsable de gestionar el procesamiento por lotes de imágenes en paralelo.
Toma una lista de rutas de imágenes y un conjunto de parámetros de procesamiento, aplica los
filtros y transformaciones seleccionados y devuelve las imágenes procesadas.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from . import filters, detect, ml, io_utils


def process_image(image_path, selected_filters, face_detection):
    """
    Aplica una serie de filtros y transformaciones a una sola imagen.

    Args:
        image_path (str): La ruta al archivo de imagen.
        selected_filters (list): Una lista de cadenas que representan los filtros a aplicar.
        face_detection (bool): Si se debe realizar la detección de rostros.

    Returns:
        tuple: Una tupla que contiene la imagen procesada y un booleano que indica
               si se detectaron rostros.
    """
    image = io_utils.load_image(image_path)

    # Apply selected filters
    for filter_name in selected_filters:
        if filter_name == "Sobel":
            image = filters.apply_sobel(image)
        elif filter_name == "Canny":
            image = filters.apply_canny(image)
        elif filter_name == "Gaussian Blur":
            image = filters.apply_gaussian_blur(image)
        elif filter_name == "Sharpen":
            image = filters.apply_sharpen(image)
        elif filter_name == "Random Hue Shift":
            image = filters.apply_random_hue_shift(image)

    # Perform face detection if enabled
    faces_detected = False
    if face_detection:
        image, faces = detect.detect_faces(image)
        if len(faces) > 0:
            faces_detected = True

    return image, faces_detected


def _process_image_wrapper(args):
    """
    Función auxiliar para desempaquetar argumentos para process_image.

    Se utiliza para permitir que `ProcessPoolExecutor` mapee un único iterable
    de tuplas de argumentos a la función `process_image`.
    """
    return process_image(*args)


def run_pipeline(image_paths, selected_filters, face_detection):
    """
    Ejecuta el pipeline de procesamiento de imágenes en paralelo.

    Args:
        image_paths (list): Una lista de rutas a los archivos de imagen.
        selected_filters (list): Una lista de cadenas que representan los filtros a aplicar.
        face_detection (bool): Si se debe realizar la detección de rostros.

    Returns:
        list: Una lista de tuplas, donde cada tupla contiene la imagen procesada
              y un booleano que indica si se detectaron rostros.
    """
    tasks = [(path, selected_filters, face_detection) for path in image_paths]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_process_image_wrapper, tasks))

    return results
