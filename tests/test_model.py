import torch
import pytest
from src.model import ImageCaptionModel

@pytest.fixture
def model():
    """Returns a dummy ImageCaptionModel."""
    return ImageCaptionModel(input_dim=768, embed_size=256, hidden_size=256, vocab_size=1000)

def test_forward_pass(model):
    """
    Tests the forward pass of the ImageCaptionModel.
    """
    # Create dummy input tensors
    batch_size = 4
    seq_length = 10
    features = torch.randn(batch_size, 768)
    captions = torch.randint(0, 1000, (batch_size, seq_length))

    # Run a forward pass
    outputs = model(features, captions)

    # Assertions
    assert outputs is not None
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, seq_length - 1, 1000)
    assert outputs.dtype == torch.float32

def test_generate_function(model):
    """
    Tests the generate function of the ImageCaptionModel.
    """
    # Create a dummy image feature tensor
    batch_size = 2
    features = torch.randn(batch_size, 768)

    # Generate captions
    generated_captions = model.generate(features)

    # Assertions
    assert generated_captions is not None
    assert isinstance(generated_captions, list)
    assert len(generated_captions) == batch_size
    for caption in generated_captions:
        assert isinstance(caption, str)
        assert len(caption) > 0
