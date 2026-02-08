"""
Script for running an app for comparison between custom and baseline model
Allows user to upload an image
"""
import os
from typing import Tuple
import yaml

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import(
    CLIPProcessor,
    CLIPVisionModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from ICGmodel import ImageCaptionModel, ModelConfig
from ICGmodel.config import CLIP_MODEL_PATH
from baseline import generate_baseline_caption

CONFIG_PATH = "src/ICGmodel/config.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/checkpoint.pth"
CAPTIONS_PATH = "data/captions.txt"

@st.cache_resource
def load_models() -> Tuple[
    CLIPProcessor,
    CLIPVisionModel,
    ImageCaptionModel,
    BlipProcessor,
    BlipForConditionalGeneration
]:
    """
    Loads all necessary models and processors, caching them to avoid
    reloading on every Streamlit rerun
    """
    # CLIP for feature extraction for custom model
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
    clip_model.eval()

    # My Custom Image Captioning Model
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = ModelConfig(
        input_dim=config["data"]["input_dim"],
        embed_size=config["model_params"]["embed_size"],
        hidden_size=config["model_params"]["hidden_size"],
        vocab_size=clip_processor.tokenizer.vocab_size,
        num_layers=config["model_params"]["lstm_layers"],
        dropout=config["training"]["dropout"]
    )

    my_model = ImageCaptionModel(model_config).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        my_model.load_state_dict(state_dict)
        my_model.eval()
    else:
        st.error(f"Error: Checkpoint not found at {CHECKPOINT_PATH}. \
                  Please train your model first.")
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # Baseline Model (BLIP)
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)

    # Return all loaded components
    return clip_processor, clip_model, my_model, blip_processor, blip_model


def generate_my_caption(
    image: Image.Image,
    processor: CLIPProcessor,
    encoder: CLIPVisionModel,
    model: ImageCaptionModel
) -> str:
    """
    Generates a caption using the custom CLIP+LSTM model
    """
    # Preprocess image using CLIP processor
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Get 768-dim features from CLIP encoder
        out = encoder(**inputs)
        features = out.pooler_output
        # Normalize features (as done in train.py)
        features = features / torch.linalg.norm(features, dim=1, keepdims=True)

        # Generate caption tokens using the custom model's beam search
        # model.generate returns List[str], so we take the first element for a single image input
        caption = model.generate(features.float(), max_len=20)[0]

    return caption

def main() -> None:
    """main script to run app"""
    st.set_page_config(page_title="Image Captioning Demo", layout="centered")
    st.title("🎓 Image Captioning Project Demo")
    st.write(
        "Upload an image to generate captions " \
        "using your custom model or a BLIP baseline."
    )

    # Show a spinner while models load
    with st.spinner("Loading models..." \
    " (this may take a minute or two depending on your connection)"):
        # Unpack the returned models and processors
        clip_processor, clip_model, my_model, blip_processor, blip_model = load_models()

    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio(
        "Choose Captioning Model:",
        ("My Custom Model (CLIP+LSTM)", "Baseline (BLIP)")
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Upload Image")

    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file).convert("RGB") # Use Image from PIL
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Generate Button
        if st.button("Generate Caption"):
            with st.spinner(f"Generating caption using {model_choice}..."):
                try:
                    if model_choice == "My Custom Model (CLIP+LSTM)":
                        caption = generate_my_caption(
                            image,
                            clip_processor,
                            clip_model,
                            my_model
                        )
                        st.success(f"**My Custom Model:** {caption}")
                    else: # Baseline (BLIP)
                        caption = generate_baseline_caption(
                            image,
                            blip_processor,
                            blip_model,
                            DEVICE
                        )
                        st.info(f"**Baseline (BLIP):** {caption}")
                except UnidentifiedImageError as e:
                    st.error(f"Cannot open image: {e}")
                except RuntimeError as e:
                    st.error(f"PyTorch runtime error: {e}")
                except ValueError as e:
                    st.error(f"Value error: {e}")
                except TypeError as e:
                    st.error(f"Type error: {e}")
    else:
        st.info("Upload an image to get started!")

if __name__ == "__main__":
    main()
