import torch
import pytest
from transformers import CLIPTokenizer
from src.model import ImageCaptionModel, ModelConfig


@pytest.fixture
def tokenizer():
    """Shared tokenizer to ensure vocab sizes match."""
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

@pytest.fixture
def dummy_model_and_checkpoint(tmp_path, tokenizer):
    """
    Creates a dummy model with the CORRECT vocab size and saves a checkpoint.
    Returns the path to the checkpoint.
    """

    config = ModelConfig(
        input_dim=768,
        embed_size=256,
        hidden_size=256,
        vocab_size=tokenizer.vocab_size,
        num_layers=1,
        dropout=0.0
    )

    model = ImageCaptionModel(config)

    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

    return model, checkpoint_path

@pytest.fixture
def model(tokenizer) -> ImageCaptionModel:
    """Returns a fresh dummy ImageCaptionModel for unit tests."""
    model_conf = ModelConfig(
        input_dim=768,
        embed_size=256,
        hidden_size=256,
        vocab_size=tokenizer.vocab_size,
        num_layers=1,
        dropout=0
    )
    return ImageCaptionModel(model_conf)


def run_inference(checkpoint_path, features, tokenizer):
    """
    A simplified inference function that mimics inference.py.
    """

    config = ModelConfig(
        input_dim=768,
        embed_size=256,
        hidden_size=256,
        vocab_size=tokenizer.vocab_size
    )

    model = ImageCaptionModel(config)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        captions = model.generate(features)

    return captions


def test_inference_produces_caption(dummy_model_and_checkpoint, tokenizer):
    """
    Tests the end-to-end inference process using the checkpoint fixture.
    """
    _, checkpoint_path = dummy_model_and_checkpoint

    features = torch.randn(1, 768)

    captions = run_inference(checkpoint_path, features, tokenizer)

    assert isinstance(captions, list)
    assert len(captions) == 1
    caption = captions[0]
    assert isinstance(caption, str)

    print(f"Generated caption: '{caption}'")
    assert len(caption) >= 0

def test_forward_pass(model: ImageCaptionModel, tokenizer):
    """
    Tests the forward pass of the ImageCaptionModel.
    """
    batch_size = 4
    seq_length = 10
    features = torch.randn(batch_size, 768)

    captions = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))

    outputs = model(features, captions)

    assert outputs is not None
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, seq_length, tokenizer.vocab_size)

def test_generate_function(model: ImageCaptionModel):
    """
    Tests the generate function (Beam Search).
    """
    batch_size = 2
    features = torch.randn(batch_size, 768)

    generated_captions = model.generate(features)

    assert generated_captions is not None
    assert isinstance(generated_captions, list)
    assert len(generated_captions) == batch_size
    for caption in generated_captions:
        assert isinstance(caption, str)