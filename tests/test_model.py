"""
tests for model forward pass
"""
import torch
import pytest
from transformers import CLIPTokenizer
from ICGmodel.model import ImageCaptionModel, ModelConfig
from ICGmodel.config import CLIP_MODEL_PATH

@pytest.fixture
def model() -> ImageCaptionModel:
    """Returns a dummy ImageCaptionModel."""

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)

    model_conf = ModelConfig(
        input_dim=768,
        embed_size=256,
        hidden_size=256,
        vocab_size=tokenizer.vocab_size,
        num_layers = 1,
        dropout = 0
    )
    return ImageCaptionModel(model_conf)

def test_forward_pass(model: ImageCaptionModel) -> None:
    """
    Tests the forward pass of the ImageCaptionModel.
    """
    # Create dummy input tensors

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)

    batch_size = 4
    seq_length = 10
    vocab_size=tokenizer.vocab_size
    features = torch.randn(batch_size, 768)
    captions = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Run a forward pass
    outputs = model(features, captions)

    # Assertions
    assert outputs is not None
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, seq_length, vocab_size)
    assert outputs.dtype == torch.float32

def test_generate_function(model:ImageCaptionModel) -> None:
    """
    Tests the generate function of the ImageCaptionModel.
    """

    # Create a dummy image feature tensor
    batch_size = 4
    features = torch.randn(batch_size, 768)

    print(f"Features shape: {features.shape}")
    # Generate captions
    generated_captions = model.generate(features)

    # Assertions
    assert generated_captions is not None
    assert isinstance(generated_captions, list)
    assert len(generated_captions) == batch_size
    for caption in generated_captions:
        assert isinstance(caption, str)
        assert len(caption) > 0
