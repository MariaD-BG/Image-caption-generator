"""
common fixtures shared across different tests
"""
import pytest
from transformers import CLIPTokenizer
from ICGmodel.config import CLIP_MODEL_PATH

@pytest.fixture
def tokenizer() -> CLIPTokenizer:
    """
    fixture for tokenizer
    """
    return CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)
