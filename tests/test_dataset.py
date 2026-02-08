import pytest
import torch
import os
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from ICGmodel.dataset import ImageDataset, collate_fn
from ICGmodel.config import CLIP_MODEL_PATH


@pytest.fixture
def dataset_resources(tmp_path):
    """
    Creates dummy data files required to initialize ImageDataset.
    We create 20 samples to verify the 80/10/10 split logic.
    """

    num_samples = 20
    feature_dim = 10

    features = {
        f"img_{i}.jpg": torch.randn(feature_dim)
        for i in range(num_samples)
    }

    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)

    captions_path = tmp_path / "captions.txt"
    with open(captions_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):

            line = f"img_{i}.jpg, Start! This is caption {i}...\n"
            f.write(line)

    return str(features_path), str(captions_path)


def test_dataset_init_and_splits(dataset_resources):
    """
    Tests initialization logic, including:
    1. Validating split arguments.
    2. Loading files correctly.
    3. The 80/10/10 split math.
    """
    feat_path, cap_path = dataset_resources

    with pytest.raises(ValueError):
        ImageDataset(feat_path, cap_path, split="invalid_split_name")

    train_ds = ImageDataset(feat_path, cap_path, split="train")
    assert len(train_ds.data) == 16

    _, first_cap_tokens = train_ds[0]

    test_ds = ImageDataset(feat_path, cap_path, split="test")
    assert len(test_ds.data) == 2


    valid_ds = ImageDataset(feat_path, cap_path, split="validation")
    assert len(valid_ds.data) == 2


def test_dataset_len(dataset_resources):
    """
    Tests the __len__ method directly.
    """
    feat_path, cap_path = dataset_resources
    dataset = ImageDataset(feat_path, cap_path, split="train")


    assert len(dataset) == 16
    assert isinstance(len(dataset), int)


def test_dataset_getitem(dataset_resources):
    """
    Tests fetching a single item.
    Verifies:
    1. It returns a Tuple.
    2. Image is a Tensor of correct shape.
    3. Caption is a Tokenized Tensor of shape (77,).
    """
    feat_path, cap_path = dataset_resources
    dataset = ImageDataset(feat_path, cap_path, split="train")

    img_tensor, cap_tokens = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (10,)

    assert isinstance(cap_tokens, torch.Tensor)
    assert cap_tokens.shape == (77,)
    assert cap_tokens.dtype == torch.long


def test_collate_fn(dataset_resources):
    """
    Tests the batch collation logic.
    We simulate a DataLoader usage to ensure batches stack correctly.
    """
    feat_path, cap_path = dataset_resources
    dataset = ImageDataset(feat_path, cap_path, split="train")

    batch = [dataset[i] for i in range(4)]

    images_batch, captions_batch = collate_fn(batch)

    assert isinstance(images_batch, torch.Tensor)
    assert isinstance(captions_batch, torch.Tensor)

    assert images_batch.shape == (4, 10)
    assert captions_batch.shape == (4, 77)