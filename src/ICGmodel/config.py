import os

CWD_PATH = os.path.join(os.getcwd(), "local_clip_model")

THIS_FILE = os.path.abspath(__file__)
REL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
REL_PATH = os.path.join(REL_ROOT, "local_clip_model")

if os.path.exists(CWD_PATH):
    CLIP_MODEL_PATH = CWD_PATH
elif os.path.exists(REL_PATH):
    CLIP_MODEL_PATH = REL_PATH
else:
    DOCKER_FALLBACK = "/project/local_clip_model"
    if os.path.exists(DOCKER_FALLBACK):
        CLIP_MODEL_PATH = DOCKER_FALLBACK
    else:
        raise FileNotFoundError(
            f"Could not find 'local_clip_model'.\n"
            f"Checked CWD: {CWD_PATH}\n"
            f"Checked Relative: {REL_PATH}\n"
            f"Checked Docker: {DOCKER_FALLBACK}"
        )

# the above was to make the docker work; the one below should also work locally

# import os

# ICG_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# SRC_DIR = os.path.dirname(ICG_MODEL_DIR)

# PROJECT_ROOT = os.path.dirname(SRC_DIR)

# CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "local_clip_model")

# print(f"DEBUG: Config is looking for model at: {CLIP_MODEL_PATH}")

# if not os.path.exists(CLIP_MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at {CLIP_MODEL_PATH}. Check your folder structure!")