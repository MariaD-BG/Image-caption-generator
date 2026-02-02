import torch

def generate_baseline_caption(image, processor, model, device):
    """
    Generates a caption for the given image using the BLIP model.

    Args:
        image: PIL Image object.
        processor: BLIP processor for image preprocessing.
        model: BLIP model for conditional generation.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        str: The generated caption.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)
