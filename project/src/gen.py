"""
Generator functions for image processing.

This module provides generator functions that are useful for image processing,
such as yielding tiles of an image for block-based processing.
"""


def tiles(image, tile_width, tile_height):
    """
    A generator that yields tiles of an image.

    Args:
        image (numpy.ndarray): The input image.
        tile_width (int): The width of the tiles.
        tile_height (int): The height of the tiles.

    Yields:
        numpy.ndarray: The next tile of the image.
    """
    for y in range(0, image.shape[0], tile_height):
        for x in range(0, image.shape[1], tile_width):
            yield image[y : y + tile_height, x : x + tile_width]
