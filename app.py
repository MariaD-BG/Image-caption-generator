import os

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, BlipProcessor, BlipForConditionalGeneration

from src import ImageCaptionModel
from baseline import generate_baseline_caption # Assuming baseline.py exists with this function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/checkpoint_trained.pth"
CAPTIONS_PATH = "data/captions.txt" # Not directly used for vocab in app.py anymore, but kept for context if needed
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    """
    Loads all necessary models and processors, caching them to avoid
    reloading on every Streamlit rerun.
    """
    # 1. CLIP for feature extraction for custom model
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_model.eval()

    # 2. My Custom Image Captioning Model
    # Parameters must match those used during training (from src/config.yaml)
    my_model = ImageCaptionModel(
        input_dim=768,
        embed_size=512, # Aligned with src/config.yaml
        hidden_size=512, # Aligned with src/config.yaml
        vocab_size=clip_processor.tokenizer.vocab_size, # Use CLIP's tokenizer vocab size
        dropout=0.2 # Aligned with src/config.yaml
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        my_model.load_state_dict(state_dict)
        my_model.eval()
    else:
        st.error(f"Error: Checkpoint not found at {CHECKPOINT_PATH}. Please train your model first.")
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # 3. Baseline Model (BLIP)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    # Return all loaded components
    return clip_processor, clip_model, my_model, blip_processor, blip_model

# --- HELPER FUNCTIONS FOR CAPTION GENERATION ---
def generate_my_caption(image, processor, encoder, model):
    """
    Generates a caption using the custom CLIP+LSTM model.
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

# --- MAIN STREAMLIT APP UI ---
def main():
    st.set_page_config(page_title="Image Captioning Demo", layout="centered")
    st.title("🎓 Image Captioning Project Demo")
    st.write("Upload an image to generate captions using your custom model or a BLIP baseline.")

    # Show a spinner while models load
    with st.spinner("Loading models... (this may take a minute or two depending on your connection)"):
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
                        caption = generate_my_caption(image, clip_processor, clip_model, my_model)
                        st.success(f"**My Custom Model:** {caption}")
                    else: # Baseline (BLIP)
                        # The generate_baseline_caption function is in baseline.py
                        caption = generate_baseline_caption(image, blip_processor, blip_model, DEVICE)
                        st.info(f"**Baseline (BLIP):** {caption}")
                except Exception as e:
                    st.error(f"Error during caption generation: {e}")
                    st.exception(e) # Display full exception for debugging
    else:
        st.info("Upload an image to get started!")

if __name__ == "__main__":
    main()
