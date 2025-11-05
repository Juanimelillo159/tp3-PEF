"""
Modelos de aprendizaje automático para la clasificación de imágenes.

Este módulo proporciona funciones para clasificar imágenes utilizando modelos de
aprendizaje automático pre-entrenados. Actualmente utiliza el modelo CLIP de OpenAI
para clasificar imágenes basándose en un conjunto de etiquetas de texto personalizadas.
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
    Clasifica un lote de imágenes utilizando el modelo CLIP.

    Args:
        images (list): Una lista de imágenes como arrays de numpy.

    Returns:
        list: Una lista de listas, donde cada lista interna contiene la etiqueta
              de clasificación superior para la imagen correspondiente.
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
