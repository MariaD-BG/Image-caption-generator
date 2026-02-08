import pytest
import torch
import yaml
import os
import argparse
from pathlib import Path
from typing import Dict

from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from eval import evaluate, bleu_test, main
from ICGmodel.model import ImageCaptionModel, ModelConfig
from ICGmodel.dataset import ImageDataset, collate_fn
from ICGmodel.config import CLIP_MODEL_PATH

@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: CLIPTokenizer) -> Dict[str, str]:
    """
    Sets up a temporary workspace that mimics your project structure.
    Using monkeypatch.chdir() ensures 'data/features.pt' in eval.py
    points to this temporary folder, not your real data.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    config = {
        "data": {
            "input_dim": 10,
            "folder_path": "data/"  # Relative path for the test execution
        },
        "model_params": {
            "embed_size": 16,     # Small sizes for speed
            "hidden_size": 16,
            "lstm_layers": 1
        },
        "training": {
            "batch_size": 2,
            "dropout": 0.0
        }
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create 10 dummy samples
    features = {str(i): torch.randn(10) for i in range(10)}
    torch.save(features, data_dir / "features.pt")

    captions_file = data_dir / "captions.txt"
    with open(captions_file, "w") as f:
        f.write("image_id,caption\n")
        for i in range(10):
            f.write(f"{i},start this is a test caption end\n")

    dummy_model_config = ModelConfig(
        input_dim=10,
        embed_size=16,
        hidden_size=16,
        vocab_size=tokenizer.vocab_size,
        num_layers=1
    )
    model = ImageCaptionModel(dummy_model_config)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    monkeypatch.chdir(tmp_path)

    return {
        "config_path": str(config_path),
        "ckpt_path": str(checkpoint_path),
        "data_dir": str(data_dir)
    }

@pytest.fixture
def tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)

@pytest.fixture
def test_loader(workspace: Dict[str, str]) -> DataLoader:
    dataset = ImageDataset(
        features_path="data/features.pt",
        cap_path="data/captions.txt",
        split="train"
    )
    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

@pytest.fixture
def tiny_model(tokenizer: CLIPTokenizer) -> ImageCaptionModel:
    config = ModelConfig(
        input_dim=10,
        embed_size=16,
        hidden_size=16,
        vocab_size=tokenizer.vocab_size,
        num_layers=1
    )
    return ImageCaptionModel(config)

def test_evaluate_function(tiny_model: ImageCaptionModel, test_loader: DataLoader) -> None:
    """
    Tests the evaluate() function in isolation.
    """
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    loss = evaluate(tiny_model, test_loader, criterion, device)

    assert isinstance(loss, float)
    assert loss >= 0

def test_bleu_function(tiny_model: ImageCaptionModel, test_loader: DataLoader, tokenizer: CLIPTokenizer) -> None:
    """
    Tests the bleu_test() function in isolation.
    """
    device = torch.device("cpu")

    score = bleu_test(tiny_model, test_loader, tokenizer, device)

    assert isinstance(score, torch.Tensor)
    assert isinstance(score.item(), float)
    assert 0.0 <= score <= 1.0

def test_eval_script_main(workspace: Dict[str, str]) -> None:
    """
    Tests the main() function.
    This covers the file loading, argument parsing logic, and the full pipeline.
    """
    args = argparse.Namespace(checkpoint=workspace['ckpt_path'])

    try:
        main(args, workspace['config_path'])
    except SystemExit:
        pytest.fail("main() called sys.exit(), which usually means a crash.")
    except Exception as e:
        pytest.fail(f"main() crashed with error: {e}")