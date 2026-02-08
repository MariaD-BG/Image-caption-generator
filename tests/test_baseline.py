import pytest
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from baseline import generate_baseline_caption

@pytest.fixture(scope="module")
def blip_resources():
    """
    Loads the BLIP model and processor.
    Scope='module' ensures we only load/download heavy model once
    per test file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading BLIP Model (this might download ~1GB if not cached)...")

    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

    return processor, model, device

@pytest.fixture
def real_image(tmp_path):
    """
    Creates a real, physical image file for testing.
    """
    img_path = tmp_path / "test_blip.jpg"
    Image.new('RGB', (224, 224), color='blue').save(img_path)
    return Image.open(img_path).convert("RGB")

def test_generate_baseline_caption_real(blip_resources, real_image):
    """
    Tests the baseline generation with REAL model inference.
    """
    processor, model, device = blip_resources

    print("\nGenerating baseline caption...")

    caption = generate_baseline_caption(
        image=real_image,
        processor=processor,
        model=model,
        device=device
    )

    print(f"BLIP Output: {caption}")

    assert isinstance(caption, str)
    assert len(caption) > 0