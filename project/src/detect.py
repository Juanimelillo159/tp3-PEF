"""
Object detection and color analysis functions.

This module provides functions for detecting objects in images, such as faces,
and for analyzing the color composition of images, such as finding the
dominant colors.
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
    Detects faces in an image using a Haar cascade.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the image with rectangles drawn around the
               detected faces and a list of the detected faces.
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
    Finds the dominant colors in an image using k-means clustering.

    Args:
        image (numpy.ndarray): The input image.
        k (int, optional): The number of dominant colors to find. Defaults to 5.

    Returns:
        numpy.ndarray: An array of the dominant colors.
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the dominant colors
    colors = kmeans.cluster_centers_.astype(int)

    return colors
