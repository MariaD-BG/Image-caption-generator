import torch
import time
import numpy as np
from PIL import Image
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
def extract_clip_features(image_path : str) -> np.ndarray:
    image = Image.open(image_path)

    # Process image to match CLIP's requirements
    inputs = processor(images=image, return_tensors="pt")

    # Get the image features; no need to compute gradients since not training yet
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.pooler_output is the feature vector (Shape: 1, 768)
    feature_vector = outputs.pooler_output.numpy().flatten()
    return feature_vector # note: this is not normalized

if __name__ == "__main__":
    features = extract_clip_features("data/bobche.jpg")
    print(f"Vector shape: {features.shape}") # Likely (768,)
    print(features)
    print(np.linalg.norm(features))