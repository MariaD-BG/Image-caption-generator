"""
Inference script
"""
import argparse
import os
import yaml

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer

from ICGmodel.model import ImageCaptionModel, ModelConfig

def get_clip_features(
        image_path : str,
        processor : CLIPProcessor,
        vision_model : CLIPVisionModel,
        device : torch.device
    ):
    """
    extracts the raw 768-dim backbone features from CLIP ViT-Base.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Image file not found: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # CLIPVisionModel outputs the raw transformer features
        outputs = vision_model(**inputs)

        # 'pooler_output' is the (1, 768) vector from the CLS token
        feature_vector = outputs.pooler_output

        # Normalize.
        features_norm = feature_vector / torch.linalg.norm(feature_vector, dim=1, keepdims=True)

    return features_norm.float()

def main(args : argparse.Namespace, config_path : str, model_name : str) -> None:
    """
    main function for inference of caption based on path to image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size

    # Load CLIP (Vision Model Only)
    print(f"Loading CLIP backbone: {model_name}...")

    processor = CLIPProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
    vision_model.eval()

    # Load Caption Model
    print("Loading Caption Model...")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = ModelConfig(
        input_dim=config["data"]["input_dim"],
        embed_size=config["model_params"]["embed_size"],
        hidden_size=config["model_params"]["hidden_size"],
        vocab_size=vocab_size,
        num_layers=config["model_params"]["lstm_layers"],
        dropout=config["training"]["dropout"]
    )

    model = ImageCaptionModel(model_config).to(device)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        print("Checkpoint loaded.")
    else:
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    model.eval()

    # Run Inference
    print(f"Processing image: {args.image}")
    features = get_clip_features(args.image, processor, vision_model, device)

    if features is None:
        return

    if features.shape[1] != config["data"]["input_dim"]:
        raise RuntimeError(f"ERROR: Expected 768 dimensions, got {features.shape[1]}")

    # Generate
    caption = model.generate(features, max_len=20)[0]

    print("\n" + "="*30)
    print(f"RESULT: {caption}")
    print("="*30 + "\n")

if __name__ == "__main__":

    CONFIG_PATH = "src/ICGmodel/config.yaml"
    MODEL_NAME = "openai/clip-vit-base-patch32"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pth")
    args_parsed = parser.parse_args()

    main(args=args_parsed, config_path=CONFIG_PATH, model_name=MODEL_NAME)
