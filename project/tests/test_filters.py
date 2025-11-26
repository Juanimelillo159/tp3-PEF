import cv2
import pytest
import numpy as np
from pathlib import Path
from project.src import filters
from project.src import io_utils

TEST_IMAGE_PATH = Path("project/data/test_image.png")
GOLDEN_IMAGE_DIR = Path("project/data/golden_images")
GOLDEN_IMAGE_DIR.mkdir(exist_ok=True)


def test_apply_sobel():
    """Tests the Sobel filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_sobel(image)

    assert isinstance(filtered_image, np.ndarray)
    assert filtered_image.shape == image.shape
    assert filtered_image.dtype == image.dtype

    golden_image_path = GOLDEN_IMAGE_DIR / "sobel.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)

    assert np.array_equal(filtered_image, golden_image)

    filtered_image_2 = filters.apply_sobel(image)
    assert np.array_equal(filtered_image, filtered_image_2)


def test_apply_random_hue_shift():
    """Tests the random hue shift filter."""
    image = io_utils.load_image(Path("project/data/color_test_image.png"))
    filtered_image = filters.apply_random_hue_shift(image)

    assert filtered_image.shape == image.shape
    assert filtered_image.dtype == image.dtype

    assert not np.array_equal(image, filtered_image)


def test_apply_canny():
    """Tests the Canny filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_canny(image)

    assert isinstance(filtered_image, np.ndarray)
    assert filtered_image.shape == image.shape
    assert filtered_image.dtype == image.dtype

    golden_image_path = GOLDEN_IMAGE_DIR / "canny.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert np.array_equal(filtered_image, golden_image)

    filtered_image_2 = filters.apply_canny(image)
    assert np.array_equal(filtered_image, filtered_image_2)


def test_apply_gaussian_blur():
    """Tests the Gaussian blur filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_gaussian_blur(image)

    assert isinstance(filtered_image, np.ndarray)
    assert filtered_image.shape == image.shape
    assert filtered_image.dtype == image.dtype

    golden_image_path = GOLDEN_IMAGE_DIR / "gaussian_blur.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert np.array_equal(filtered_image, golden_image)

    filtered_image_2 = filters.apply_gaussian_blur(image)
    assert np.array_equal(filtered_image, filtered_image_2)


def test_apply_sharpen():
    """Tests the sharpen filter."""
    image = io_utils.load_image(TEST_IMAGE_PATH)
    filtered_image = filters.apply_sharpen(image)

    assert isinstance(filtered_image, np.ndarray)
    assert filtered_image.shape == image.shape
    assert filtered_image.dtype == image.dtype

    golden_image_path = GOLDEN_IMAGE_DIR / "sharpen.png"

    if not golden_image_path.exists():
        io_utils.save_image(filtered_image, golden_image_path)

    golden_image = io_utils.load_image(golden_image_path)
    assert np.array_equal(filtered_image, golden_image)

    filtered_image_2 = filters.apply_sharpen(image)
    assert np.array_equal(filtered_image, filtered_image_2)
