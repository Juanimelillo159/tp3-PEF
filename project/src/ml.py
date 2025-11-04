"""
Machine learning models for image classification.

This module provides functions for classifying images using pre-trained
machine learning models. It currently uses the CLIP model from OpenAI
to classify images based on a set of custom text labels.
"""
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the custom labels
LABELS = [
    "a photo of a person",
    "a photo of a building",
    "an abstract drawing",
    "other",
]


def classify_batch(images):
    """
    Classifies a batch of images using the CLIP model.

    Args:
        images (list): A list of images as numpy arrays.

    Returns:
        list: A list of lists, where each inner list contains the top
              classification label for the corresponding image.
    """

    # Preprocess the images and labels
    inputs = processor(
        text=LABELS,
        images=[Image.fromarray(image) for image in images],
        return_tensors="pt",
        padding=True,
    )

    # Make predictions
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

    # Get the top prediction for each image
    probs = logits_per_image.softmax(dim=1)
    top_predictions = probs.argmax(dim=1)

    # Decode the predictions
    results = [[LABELS[prediction.item()]] for prediction in top_predictions]

    return results
