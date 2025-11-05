"""
Funciones generadoras para el procesamiento de imágenes.

Este módulo proporciona funciones generadoras que son útiles para el procesamiento de imágenes,
como producir teselas de una imagen para el procesamiento basado en bloques.
"""


def tiles(image, tile_width, tile_height):
    """
    Un generador que produce teselas de una imagen.

    Args:
        image (numpy.ndarray): La imagen de entrada.
        tile_width (int): El ancho de las teselas.
        tile_height (int): La altura de las teselas.

    Yields:
        numpy.ndarray: La siguiente tesela de la imagen.
    """
    for y in range(0, image.shape[0], tile_height):
        for x in range(0, image.shape[1], tile_width):
            yield image[y : y + tile_height, x : x + tile_width]
