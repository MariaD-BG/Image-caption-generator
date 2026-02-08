import pytest
import torch
import os
import yaml
import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer

from inference import get_clip_features, main
from ICGmodel.model import ImageCaptionModel, ModelConfig
from ICGmodel.config import CLIP_MODEL_PATH


@pytest.fixture
def tokenizer():
    """Load the real local tokenizer."""
    return CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)

@pytest.fixture
def processor():
    """Load the real local processor."""
    return CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

@pytest.fixture
def vision_model():
    """Load the real local vision model."""
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL_PATH)
    model.eval()
    return model

@pytest.fixture
def workspace(tmp_path, tokenizer):
    """
    Sets up a full dummy environment:
    1. A dummy image.
    2. A dummy config (WITH input_dim=512 to match CLIP).
    3. A dummy checkpoint file.
    """
    # Create Dummy Image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test_image.jpg"
    # Create a small red square image
    Image.new('RGB', (224, 224), color='red').save(img_path)

    config = {
        "data": {
            "input_dim": 768,
            "folder_path": "data/"
        },
        "model_params": {
            "embed_size": 16,
            "hidden_size": 16,
            "lstm_layers": 1
        },
        "training": {
            "dropout": 0.0
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # We must create a model that matches the config above so it can be saved/loaded
    model_config = ModelConfig(
        input_dim=768,
        embed_size=16,
        hidden_size=16,
        vocab_size=tokenizer.vocab_size,
        num_layers=1
    )
    model = ImageCaptionModel(model_config)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "checkpoint.pth"

    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    return {
        "image_path": str(img_path),
        "config_path": str(config_path),
        "ckpt_path": str(ckpt_path),
        "input_dim": 512
    }


def test_get_clip_ok(workspace, processor, vision_model):
    """
    Test that features are extracted correctly from a real image.
    """
    device = torch.device("cpu")
    features = get_clip_features(workspace["image_path"], processor, vision_model, device)

    assert features is not None
    assert torch.is_tensor(features)
    assert len(features.shape) == 2
    assert features.shape[0] == 1
    assert features.shape[1] == 768

def test_get_clip_missing_f(tmp_path, processor, vision_model, capsys):
    """
    Test behavior when the image file does not exist.
    """
    device = torch.device("cpu")
    bad_path = str(tmp_path / "ghost.jpg")

    features = get_clip_features(bad_path, processor, vision_model, device)

    assert features is None
    captured = capsys.readouterr()
    assert "Image file not found" in captured.out

def test_inference_main(workspace):
    """
    Test the full main() pipeline with valid arguments.
    """
    args = argparse.Namespace(
        image=workspace["image_path"],
        checkpoint=workspace["ckpt_path"]
    )

    try:
        main(args, workspace["config_path"], model_name=CLIP_MODEL_PATH)
    except Exception as e:
        pytest.fail(f"main() raised an exception unexpectedly: {e}")
