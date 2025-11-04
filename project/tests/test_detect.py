import cv2
import pytest
from pathlib import Path
from project.src import detect
from project.src import io_utils

# Define the paths for the test image and golden images
TEST_IMAGE_PATH = Path("project/data/test_image.png")


def test_detect_faces():
    """Tests the face detection."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    _, faces = detect.detect_faces(image)

    # In this simple test image, we don't expect any faces
    assert len(faces) == 0


def test_get_dominant_colors():
    """Tests the dominant color extraction."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    colors = detect.get_dominant_colors(image, k=2)

    # We expect two dominant colors: black and white
    # Note: The order is not guaranteed
    assert ([0, 0, 0] in colors) and ([255, 255, 255] in colors)
