import pytest
import torch
import os
import yaml
from PIL import Image
from unittest.mock import MagicMock, patch

import app
from ICGmodel.config import CLIP_MODEL_PATH
from ICGmodel.model import ModelConfig, ImageCaptionModel


@pytest.fixture
def real_image(tmp_path):
    """Creates a real physical image file for testing."""
    img_path = tmp_path / "test_image.jpg"
    Image.new('RGB', (224, 224), color='red').save(img_path)
    return Image.open(img_path).convert("RGB")

@pytest.fixture
def setup_real_paths(tmp_path):
    """
    Sets up a valid config and checkpoint so load_models() doesn't crash.
    We point the app's global variables to these temp files.
    """

    config_data = {
        "data": {"input_dim": 768},
        "model_params": {
            "embed_size": 16,
            "hidden_size": 16,
            "lstm_layers": 1
        },
        "training": {"dropout": 0.0}
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    dummy_config = ModelConfig(
        input_dim=768,
        embed_size=16,
        hidden_size=16,
        vocab_size=tokenizer.vocab_size,
        num_layers=1
    )
    model = ImageCaptionModel(dummy_config)

    ckpt_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    return str(config_path), str(ckpt_path)

# --- THE TESTS ---

def test_load_models_real(setup_real_paths):
    """
    1. Test load_models() with REAL weights.
    We patch 'st.cache_resource' because it fails outside of a running Streamlit server,
    but we let the inner function run fully.
    """
    config_path, ckpt_path = setup_real_paths

    # Temporarily override the hardcoded paths in app.py with our test paths
    # (So it doesn't fail if you haven't trained a model yet)
    with patch("app.CONFIG_PATH", config_path), \
         patch("app.CHECKPOINT_PATH", ckpt_path), \
         patch("streamlit.cache_resource", side_effect=lambda func: func): # Bypass cache decorator

        print("\nLoading REAL models... (This will be slow)...")
        components = app.load_models()

        # Unpack to verify types
        clip_proc, clip_vis, my_model, blip_proc, blip_gen = components

        # Verify we got real objects, not mocks
        assert isinstance(clip_proc, app.CLIPProcessor)
        assert isinstance(clip_vis, app.CLIPVisionModel)
        assert isinstance(my_model, app.ImageCaptionModel)
        # Check if the model is on the correct device
        assert str(next(my_model.parameters()).device) == str(app.DEVICE)

def test_generate_my_caption_real(real_image, setup_real_paths):
    """
    2. Test generate_my_caption() with REAL execution.
    This runs the image through CLIP and your LSTM.
    """
    config_path, ckpt_path = setup_real_paths

    with patch("app.CONFIG_PATH", config_path), \
         patch("app.CHECKPOINT_PATH", ckpt_path), \
         patch("streamlit.cache_resource", side_effect=lambda func: func):

        # Load the real tools
        clip_proc, clip_vis, my_model, _, _ = app.load_models()

        print("\nGenerating caption... (Running forward pass)...")
        caption = app.generate_my_caption(real_image, clip_proc, clip_vis, my_model)

        print(f"Generated: {caption}")

        # Basic sanity checks
        assert isinstance(caption, str)
        # Even an untrained model should output something (start/end tokens or random words)
        # This confirms the tensor shapes matched and the forward pass succeeded.

@patch("app.st") # We MUST mock Streamlit UI elements (buttons/sidebars) or the test hangs
def test_main_real_execution(mock_st, real_image, setup_real_paths):
    """
    3. Test main() End-to-End.
    We Mock the UI inputs (User Clicks), but we execute the REAL logic.
    """
    config_path, ckpt_path = setup_real_paths

    with patch("app.CONFIG_PATH", config_path), \
         patch("app.CHECKPOINT_PATH", ckpt_path), \
         patch("streamlit.cache_resource", side_effect=lambda func: func):

        # --- SIMULATE USER INPUT ---
        # 1. User selects "My Custom Model"
        mock_st.sidebar.radio.return_value = "My Custom Model (CLIP+LSTM)"

        # 2. User uploads a file
        # We need to simulate the uploaded file object.
        # Streamlit returns a file-like object, so we mock that slightly to work with Image.open
        # However, app.py calls Image.open(uploaded_file).
        # We can just patch Image.open to return our real_image fixture when called inside main.
        with patch("app.Image.open", return_value=real_image):
            # The uploader itself just needs to return *something* not None
            mock_st.sidebar.file_uploader.return_value = "dummy_filename.jpg"

            # 3. User clicks "Generate Caption"
            mock_st.button.return_value = True

            print("\nRunning Main App Logic...")
            app.main()

            # --- VERIFY RESULTS ---
            # Check if success was called.
            # The args[0] of the call will be the string "**My Custom Model:** <caption_text>"
            assert mock_st.success.called
            args, _ = mock_st.success.call_args
            assert "**My Custom Model:**" in args[0]
            print(f"UI Output: {args[0]}")