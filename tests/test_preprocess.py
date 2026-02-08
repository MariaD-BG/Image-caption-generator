import pytest
import torch
from PIL import Image
from pathlib import PosixPath
from typing import List, Tuple

from transformers import CLIPConfig, CLIPVisionModel, CLIPProcessor

from preprocess import calc_and_save

@pytest.fixture
def tiny_model() -> Tuple[CLIPVisionModel, CLIPProcessor]:
    """
    Creates a REAL CLIP model and processor, but initialized with a tiny
    configuration so it runs instantly and doesn't need to download files.
    """

    config = CLIPConfig(
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "projection_dim": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 224,
            "patch_size": 10
        }
    )

    model = CLIPVisionModel(config.vision_config)
    model.eval()

    processor = CLIPProcessor(
        image_size=224,
        do_resize=True,
        do_center_crop=True
    )

    return model, processor

@pytest.fixture
def real_images(tmp_path : PosixPath) -> Tuple[PosixPath, List[str]]:
    """
    Creates a temporary folder with real (but random) JPEG images.
    """
    img_folder = tmp_path / "Images"
    img_folder.mkdir()

    filenames = ["test_img_A.jpg", "test_img_B.jpg"]

    for fname in filenames:
        img = Image.new('RGB', (100, 100), color=(150, 50, 50))
        img.save(img_folder / fname)

    return img_folder, filenames

def test_calc_and_save(
        tiny_model : Tuple[CLIPVisionModel, CLIPProcessor],
        real_images: Tuple[PosixPath, List[str]]
) -> None:
    """
    Tests the full pipeline using real objects.
    """
    model, processor = tiny_model
    img_folder, filenames = real_images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dict = calc_and_save(
        model=model,
        img_folder=str(img_folder),
        img_names=filenames,
        processor=processor,
        device=device
    )

    assert isinstance(features_dict, dict)
    assert len(features_dict) == 2
    assert "test_img_A.jpg" in features_dict

    tensor_a = features_dict["test_img_A.jpg"]
    assert isinstance(tensor_a, torch.Tensor)
    assert tensor_a.shape == (32,)

    # 3. Check Logic: Is it normalized?
    # The code divides by the norm, so the resulting norm must be 1.0
    norm = torch.linalg.norm(tensor_a)

    # We use a small tolerance (atol) because of floating point math
    assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5), \
    f"Feature vector is not normalized! Norm is {norm}"

    print(f"\nSuccess! Generated vector shape: {tensor_a.shape}, Norm: {norm}")

def test_device_movement(
        tiny_model : Tuple[CLIPVisionModel, CLIPProcessor],
        real_images: Tuple[PosixPath, List[str]]
) -> None:
    """
    Verifies that the code correctly moves inputs to the specified device.
    """
    model, processor = tiny_model
    img_folder, filenames = real_images

    # If you have a GPU available on the cluster, you can change this to "cuda"
    # Otherwise, we stick to CPU to ensure the test passes everywhere.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_dict = calc_and_save(
        model=model,
        img_folder=str(img_folder),
        img_names=filenames,
        processor=processor,
        device=device
    )

    assert features_dict["test_img_A.jpg"].device.type == "cpu"