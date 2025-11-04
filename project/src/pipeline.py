import os
from concurrent.futures import ProcessPoolExecutor
from . import filters, detect, ml, io_utils

def process_image(image_path, selected_filters, face_detection):
    """Processes a single image."""
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

    # Perform face detection if enabled
    if face_detection:
        image, _ = detect.detect_faces(image)

    return image

def _process_image_wrapper(args):
    """Helper function to unpack arguments for process_image."""
    return process_image(*args)

def run_pipeline(image_paths, selected_filters, face_detection):
    """Runs the image processing pipeline in parallel."""
    tasks = [(path, selected_filters, face_detection) for path in image_paths]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_process_image_wrapper, tasks))

    return results
