"""
tests for training script
"""
import os
from pathlib import PosixPath
from typing import Dict, Union

import pytest
import torch
import yaml
from transformers import CLIPTokenizer

from train import train_model, save_checkpoint
from ICGmodel import ImageCaptionModel, ModelConfig


@pytest.fixture
def workspace(tmp_path : PosixPath) -> Dict[str, Union[str, int]]:
    """
    Creates a temporary workspace with:
    1. A dummy config.yaml
    2. A dummy features.pt
    3. A dummy captions.txt
    4. Folders for saving plots/checkpoints
    """

    data_dir = tmp_path / "data"
    plots_dir = tmp_path / "plots"
    ckpt_dir = tmp_path / "checkpoints"

    data_dir.mkdir()
    plots_dir.mkdir()
    ckpt_dir.mkdir()

    config = {
        "data": {
            "input_dim": 10,
            "folder_path": str(data_dir) + "/"
        },
        "model_params": {
            "embed_size": 4,
            "hidden_size": 4,
            "lstm_layers": 1
        },
        "training": {
            "batch_size": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "epochs": 1
        },
        "saving": {
            "plots_path": str(plots_dir) + "/",
            "checkpoints_path": str(ckpt_dir) + "/"
        }
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


    features = {str(i): torch.randn(10) for i in range(10)}
    torch.save(features, data_dir / "features.pt")

    captions_file = data_dir / "captions.txt"
    with open(captions_file, "w", encoding="utf-8") as f:
        f.write("image_id,caption\n")
        for i in range(10):
            # Write a simple caption for each image
            f.write(f"{i},this is a test caption for image {i}\n")

    return {
        "config_path": str(config_path),
        "plots_path": str(plots_dir),
        "ckpt_path": str(ckpt_dir),
        "input_dim": 10
    }


def test_save_checkpoint(tmp_path : PosixPath, tokenizer : CLIPTokenizer) -> None:
    """
    Test if save_checkpoint actually creates a file on the disk
    """

    config = ModelConfig(
        input_dim=10,
        embed_size=4,
        hidden_size=4,
        vocab_size=tokenizer.vocab_size,
        num_layers=1
    )
    model = ImageCaptionModel(config)

    save_path = tmp_path / "test_checkpoint.pth"

    save_checkpoint(model, filepath=str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0

    loaded = torch.load(save_path)
    assert 'model_state_dict' in loaded

def test_train_model(workspace):
    """
    Test the full training loop with real files and real model.
    This ensures no crashes occur during data loading, forward pass, or saving.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_model(workspace['config_path'], device)

    assert isinstance(model, ImageCaptionModel)

    param = next(model.parameters())
    assert param.requires_grad is True

    saved_files = os.listdir(workspace['ckpt_path'])
    assert len(saved_files) > 0, "No checkpoint was saved!"
    assert "checkpoint.pth" in saved_files[0]
