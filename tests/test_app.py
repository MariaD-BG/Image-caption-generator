import pytest
import torch
import yaml
from PIL import Image
from unittest.mock import patch
from transformers import CLIPTokenizer

import app
from ICGmodel.model import ModelConfig, ImageCaptionModel
from ICGmodel.config import CLIP_MODEL_PATH

@pytest.fixture
def tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)

@pytest.fixture
def real_files(tmp_path, tokenizer):
    """
    Creates real files (config.yaml, checkpoint.pth, image.jpg)
    so the app logic can run without crashing.
    """

    config = {
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
        yaml.dump(config, f)

    model_config = ModelConfig(
        input_dim=768,
        embed_size=16,
        hidden_size=16,
        vocab_size=tokenizer.vocab_size,
        num_layers=1
    )
    model = ImageCaptionModel(model_config)
    ckpt_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # 3. Create a Dummy Image
    img_path = tmp_path / "test.jpg"
    Image.new('RGB', (224, 224), color='red').save(img_path)

    return str(config_path), str(ckpt_path), str(img_path)


def test_load_models(real_files):
    """
    Tests loading the REAL models from disk/internet.
    """
    config_path, ckpt_path, _ = real_files

    with patch("app.CONFIG_PATH", config_path), \
         patch("app.CHECKPOINT_PATH", ckpt_path), \
         patch("streamlit.cache_resource", side_effect=lambda func: func): # Bypass Streamlit cache

        print("\nLoading models (this may take time)...")
        components = app.load_models()

        assert len(components) == 5
        clip_proc, clip_vis, my_model, blip_proc, blip_gen = components

        assert isinstance(clip_proc, app.CLIPProcessor)
        assert isinstance(clip_vis, app.CLIPVisionModel)
        assert isinstance(my_model, app.ImageCaptionModel)
        assert isinstance(blip_proc, app.BlipProcessor)


def test_generate_my_caption(real_files):
    """
    Tests the caption generation using REAL models and REAL images.
    """
    config_path, ckpt_path, img_path = real_files
    image = Image.open(img_path).convert("RGB")

    with patch("app.CONFIG_PATH", config_path), \
         patch("app.CHECKPOINT_PATH", ckpt_path), \
         patch("streamlit.cache_resource", side_effect=lambda func: func):

        clip_proc, clip_vis, my_model, _, _ = app.load_models()

        print("\nGenerating caption...")
        caption = app.generate_my_caption(image, clip_proc, clip_vis, my_model)

        print(f"Result: {caption}")
        assert isinstance(caption, str)
        assert len(caption) > 0


@patch("app.st") # Must mock UI components
def test_main(mock_st, real_files):
    """
    Tests the main app flow.
    We mock the USER INPUT (clicks), but run the REAL LOGIC.
    """
    config_path, ckpt_path, img_path = real_files

    with patch("app.CONFIG_PATH", config_path), \
         patch("app.CHECKPOINT_PATH", ckpt_path), \
         patch("streamlit.cache_resource", side_effect=lambda func: func):

        #  Simulate User Actions
        # User selects "My Custom Model"
        mock_st.sidebar.radio.return_value = "My Custom Model (CLIP+LSTM)"

        # 2. User uploads a file (We mimic the Streamlit file buffer)
        # Streamlit returns a file-like object. We can just use the path or open file.
        # But app.py calls Image.open(uploaded_file).
        # So we trick Image.open inside main to return our real image.
        with patch("app.Image.open", return_value=Image.open(img_path)):
            mock_st.sidebar.file_uploader.return_value = "uploaded_stuff"

            mock_st.button.return_value = True

            app.main()

            assert mock_st.success.called
            args, _ = mock_st.success.call_args
            assert "**My Custom Model:**" in args[0]