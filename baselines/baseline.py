from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_baseline_caption(image, processor, model, device):
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
