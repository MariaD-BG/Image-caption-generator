"""
Script for loading processor and tokenizer CLIP models
"""
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

MODEL_NAME = "openai/clip-vit-base-patch32"
SAVE_DICT = "./local_clip_model"

print(f"Downloading {MODEL_NAME}...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

model.save_pretrained(SAVE_DICT)
processor.save_pretrained(SAVE_DICT)
tokenizer.save_pretrained(SAVE_DICT)

print(f"Model saved to {SAVE_DICT}")
