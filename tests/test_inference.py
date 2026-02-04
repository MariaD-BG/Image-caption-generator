import torch
import pytest
from src.model import ImageCaptionModel

@pytest.fixture
def dummy_model_and_checkpoint(tmp_path):
    """
    Creates a dummy model and saves a checkpoint for it.
    Returns the path to the checkpoint.
    """
    model = ImageCaptionModel(input_dim=768, embed_size=256, hidden_size=256, vocab_size=1000)
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    return model, checkpoint_path

def run_inference(checkpoint_path, features):
    """
    A simplified inference function that mimics inference.py.
    """
    model = ImageCaptionModel(input_dim=768, embed_size=256, hidden_size=256, vocab_size=1000)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        captions = model.generate(features)
    
    return captions

def test_inference_produces_caption(dummy_model_and_checkpoint):
    """
    Tests the end-to-end inference process.
    """
    _, checkpoint_path = dummy_model_and_checkpoint
    
    # Create a dummy image feature tensor
    features = torch.randn(1, 768)

    # Run the inference
    captions = run_inference(checkpoint_path, features)

    # Assertions
    assert isinstance(captions, list)
    assert len(captions) == 1
    caption = captions[0]
    assert isinstance(caption, str)
    assert len(caption) > 0
    # A simple check for a plausible sentence
    assert len(caption.split()) > 1
