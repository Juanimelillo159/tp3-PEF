import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

# Load the pre-trained MobileNetV3 model
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

# Define the image transformations
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_batch(images):
    """Classifies a batch of images using MobileNetV3."""
    # Preprocess the images
    batch = torch.stack([preprocess(Image.fromarray(image)) for image in images])

    # Make predictions
    with torch.no_grad():
        predictions = model(batch)

    # Get the top-k predictions
    top3 = torch.topk(predictions, 3)

    # Decode the predictions
    results = []
    for i in range(len(images)):
        labels = [weights.meta["categories"][j] for j in top3.indices[i]]
        results.append(labels)

    return results
