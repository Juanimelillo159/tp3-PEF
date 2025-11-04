"""
Orchestrates the image processing pipeline.

This module is responsible for managing the batch processing of images in parallel.
It takes a list of image paths and a set of processing parameters, applies the
selected filters and transformations, and returns the processed images.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from . import filters, detect, ml, io_utils


def process_image(image_path, selected_filters, face_detection):
    """
    Applies a series of filters and transformations to a single image.

    Args:
        image_path (str): The path to the image file.
        selected_filters (list): A list of strings representing the filters to apply.
        face_detection (bool): Whether to perform face detection.

    Returns:
        tuple: A tuple containing the processed image and a boolean indicating
               whether faces were detected.
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
    Helper function to unpack arguments for process_image.

    This is used to allow the `ProcessPoolExecutor` to map a single iterable
    of argument tuples to the `process_image` function.
    """
    return process_image(*args)


def run_pipeline(image_paths, selected_filters, face_detection):
    """
    Runs the image processing pipeline in parallel.

    Args:
        image_paths (list): A list of paths to the image files.
        selected_filters (list): A list of strings representing the filters to apply.
        face_detection (bool): Whether to perform face detection.

    Returns:
        list: A list of tuples, where each tuple contains the processed image
              and a boolean indicating whether faces were detected.
    """
    tasks = [(path, selected_filters, face_detection) for path in image_paths]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_process_image_wrapper, tasks))

    return results
