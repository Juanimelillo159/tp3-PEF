import cv2
import pytest
import numpy as np
from pathlib import Path
from project.src import detect
from project.src import io_utils

TEST_IMAGE_PATH = Path("project/data/test_image.png")


def test_detect_faces():
    """Tests the face detection."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    processed_image, faces = detect.detect_faces(image)

    assert isinstance(processed_image, np.ndarray)
    assert processed_image.shape == image.shape

    assert len(faces) == 0


def test_get_dominant_colors():
    """Tests the dominant color extraction."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    colors = detect.get_dominant_colors(image, k=2)

    assert isinstance(colors, np.ndarray)
    assert colors.shape == (2, 3)
    assert np.issubdtype(colors.dtype, np.integer)

    colors_list = colors.tolist()

    assert [0, 0, 0] in colors_list
    assert [255, 255, 255] in colors_list
