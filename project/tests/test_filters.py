import cv2
import pytest
from pathlib import Path
from project.src import filters
from project.src import io_utils

# Define the paths for the test image and golden images
TEST_IMAGE_PATH = Path("project/data/test_image.png")
GOLDEN_IMAGE_DIR = Path("project/data/golden_images")
GOLDEN_IMAGE_DIR.mkdir(exist_ok=True)

def test_apply_sobel():
    """Tests the Sobel filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_sobel(image)

    golden_image_path = GOLDEN_IMAGE_DIR / "sobel.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert (filtered_image == golden_image).all()

def test_apply_canny():
    """Tests the Canny filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_canny(image)

    golden_image_path = GOLDEN_IMAGE_DIR / "canny.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert (filtered_image == golden_image).all()

def test_apply_gaussian_blur():
    """Tests the Gaussian blur filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_gaussian_blur(image)

    golden_image_path = GOLDEN_IMAGE_DIR / "gaussian_blur.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert (filtered_image == golden_image).all()

def test_apply_sharpen():
    """Tests the sharpen filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_sharpen(image)

    golden_image_path = GOLDEN_IMAGE_DIR / "sharpen.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert (filtered_image == golden_image).all()
