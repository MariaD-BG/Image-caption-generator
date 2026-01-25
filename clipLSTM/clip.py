import torch
import time
import numpy as np
from PIL import Image
from typing import List
from transformers import CLIPProcessor, CLIPVisionModel

# Load CLIP (to use as encoder model)
# "openai/clip-vit-base-patch32" is the standard small (hence fast) version
start = time.time()
print("Loading model...")
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
print("Loading processor...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
end = time.time()
print(f"Processor loaded in {end-start}s")

# Preprocess and Extract Features
def extract_clip_features(batched_images : torch.tensor) -> torch.tensor:

    with torch.no_grad():
        outputs = model(pixel_values=batched_images) # Get the image features; no need to compute gradients since not training yet

    feature_vector = outputs.pooler_output # outputs.pooler_output is the feature vector; Shape: (batch_size, 768)
    features_norm = feature_vector / torch.linalg.norm(feature_vector, dim=1, keepdims=True)
    return features_norm # note: this is not normalized
