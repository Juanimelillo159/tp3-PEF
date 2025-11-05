"""
Funciones de detección de objetos y análisis de color.

Este módulo proporciona funciones para detectar objetos en imágenes, como rostros,
y para analizar la composición de color de las imágenes, como encontrar los
colores dominantes.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from .cache import memoize

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@memoize
def detect_faces(image):
    """
    Detecta rostros en una imagen usando una cascada de Haar.

    Args:
        image (numpy.ndarray): La imagen de entrada.

    Returns:
        tuple: Una tupla que contiene la imagen con rectángulos dibujados alrededor de los
               rostros detectados y una lista de los rostros detectados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the faces
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image, faces


@memoize
def get_dominant_colors(image, k=5):
    """
    Encuentra los colores dominantes en una imagen usando k-means clustering.

    Args:
        image (numpy.ndarray): La imagen de entrada.
        k (int, optional): El número de colores dominantes a encontrar. Por defecto es 5.

    Returns:
        numpy.ndarray: Un array de los colores dominantes.
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the dominant colors
    colors = kmeans.cluster_centers_.astype(int)

    return colors
