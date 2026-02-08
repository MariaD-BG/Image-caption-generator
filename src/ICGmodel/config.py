import os

ICG_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.dirname(ICG_MODEL_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)

CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "local_clip_model")

print(f"DEBUG: Config is looking for model at: {CLIP_MODEL_PATH}")

if not os.path.exists(CLIP_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {CLIP_MODEL_PATH}. Check your folder structure!")