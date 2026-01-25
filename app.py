import streamlit as st
import torch
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPVisionModel, BlipProcessor, BlipForConditionalGeneration

from clipLSTM.model import ImageCaptionModel
from clipLSTM.dataset import Vocabulary
from baselines.baseline import generate_baseline_caption

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/checkpoint.pth"
CAPTIONS_PATH = "data/captions.txt"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32" # 512 backbone, 768 output

# --- CACHING RESOURCES ---
# @st.cache_resource is CRITICAL. It loads the models only once.
# Without this, the app would reload the heavy models every time you click a button.
@st.cache_resource
def load_models():
    # 1. Load Vocabulary
    vocab = Vocabulary(freq_threshold=10)
    vocab.build_vocabulary(captions_path=CAPTIONS_PATH)

    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_model.eval()

    my_model = ImageCaptionModel(
        input_dim=768,
        embed_size=128,
        hidden_size=32,
        vocab_size=len(vocab),
        dropout=0.2
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        my_model.load_state_dict(state_dict)
        my_model.eval()
    else:
        st.error(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # 4. Load Baseline Model (BLIP)
    # This satisfies the requirement to compare with a ready-made model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    return vocab, clip_processor, clip_model, my_model, blip_processor, blip_model

# --- HELPER FUNCTIONS ---
def generate_my_caption(image, vocab, processor, encoder, model):
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Get 768-dim features
        out = encoder(**inputs)
        features = out.pooler_output
        # Normalize
        features = features / torch.linalg.norm(features, dim=1, keepdims=True)

        # Generate (ensure your model.generate has the <SOS> fix!)
        caption_tokens = model.generate(features.float(), vocab, max_len=20)

    return " ".join(caption_tokens).replace("<SOS>", "").replace("<EOS>", "").strip()

# def generate_blip_caption(image, processor, model):
#     inputs = processor(image, return_tensors="pt").to(DEVICE)
#     out = model.generate(**inputs)
#     return processor.decode(out[0], skip_special_tokens=True)

# --- MAIN APP UI ---
def main():
    st.title("🎓 Image Captioning Project")
    st.write("Upload an image to generate a caption using my custom model or a baseline.")

    # Show a spinner while models load
    with st.spinner("Loading models... (this may take a minute)"):
        vocab, clip_proc, clip_enc, my_model, blip_proc, blip_model = load_models()

    # Sidebar controls
    st.sidebar.header("Settings")
    model_choice = st.sidebar.radio(
        "Choose Model:",
        ("My Custom Model (CLIP+LSTM)", "Baseline (BLIP)")
    )

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Generate Button
        if st.button("Generate Caption"):
            with st.spinner("Analyzing image..."):
                try:
                    if model_choice == "My Custom Model (CLIP+LSTM)":
                        caption = generate_my_caption(image, vocab, clip_proc, clip_enc, my_model)
                        st.success(f"**My Model:** {caption}")
                    else:
                        caption = generate_baseline_caption(image, blip_proc, blip_model, DEVICE)
                        st.info(f"**Baseline (BLIP):** {caption}")
                except Exception as e:
                    st.error(f"Error during generation: {e}")

if __name__ == "__main__":
    main()