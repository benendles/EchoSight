import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


# Choose device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")


# Load Depth Anything V2
model_name = "depth-anything/Depth-Anything-V2-Small-hf"

processor = AutoImageProcessor.from_pretrained(model_name)

model = AutoModelForDepthEstimation.from_pretrained(
    model_name
)

model.to(device)
model.eval()

print("Depth Anything V2 loaded successfully!")


def estimate_depth(image_path):
    """
    Estimate relative depth for an image.

    Returns:
        NumPy depth map with the same spatial dimensions
        as the original image.
    """

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare input
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run depth model
    with torch.no_grad():
        outputs = model(**inputs)

    # Predicted depth
    predicted_depth = outputs.predicted_depth

    # Original image dimensions
    original_width, original_height = image.size

    # Resize depth map to original image size
    resized_depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False
    )

    # Convert to NumPy
    depth = resized_depth.squeeze().cpu().numpy()

    return depth