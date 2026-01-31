import argparse
import os

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer

from src.model import ImageCaptionModel

def get_clip_features(image_path, processor, vision_model, device):
    """
    extracts the raw 768-dim backbone features from CLIP ViT-Base.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # CLIPVisionModel outputs the raw transformer features
        outputs = vision_model(**inputs)

        # 'pooler_output' is the (1, 768) vector from the CLS token
        feature_vector = outputs.pooler_output

        # Optional: Normalize.
        features_norm = feature_vector / torch.linalg.norm(feature_vector, dim=1, keepdims=True)

    return features_norm.float()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size

    # --- 2. Load CLIP (Vision Model Only) ---
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP backbone: {model_name}...")

    processor = CLIPProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
    vision_model.eval()

    # --- 3. Load Caption Model ---
    print("Loading Caption Model...")
    # These sizes should match the saved model's config
    embed_size = 512
    hidden_size = 512
    model = ImageCaptionModel(
        input_dim=768,
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dropout=0.2
    ).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        print("Checkpoint loaded.")
    else:
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    model.eval()

    # --- 4. Run Inference ---
    print(f"Processing image: {args.image}")
    features = get_clip_features(args.image, processor, vision_model, device)

    if features is None: return

    if features.shape[1] != 768:
        raise RuntimeError(f"ERROR: Expected 768 dimensions, got {features.shape[1]}")

    # Generate
    caption = model.generate(features, max_len=20)[0]

    print("\n" + "="*30)
    print(f"RESULT: {caption}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pth")
    args = parser.parse_args()
    main(args)
