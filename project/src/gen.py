def tiles(image, tile_width, tile_height):
    """A generator that yields tiles of an image."""
    for y in range(0, image.shape[0], tile_height):
        for x in range(0, image.shape[1], tile_width):
            yield image[y:y + tile_height, x:x + tile_width]
