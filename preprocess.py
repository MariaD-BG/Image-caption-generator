"""
Preprocess raw CLIP images and save features to a dict
"""
import time
from pathlib import Path
from typing import List, Dict

import torch
import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

def calc_and_save(
        model: CLIPVisionModel,
        img_folder: str,
        img_names: List[str],
        processor: CLIPProcessor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
    """
    Calculate features
    """

    images = [Image.open(Path(img_folder) / img_name).convert("RGB") for img_name in img_names]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model(**inputs).pooler_output.cpu()

    features_norm = features / torch.linalg.norm(features, dim=1, keepdims=True)

    features_dict = dict(zip(img_names, features_norm))

    return features_dict

def main(model_name : str, batch_size: int) -> None:
    """
    Main method for loading model and calculating and saving features
    """
    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name).to(device)
    model.eval()

    data_folder = Path("data")
    img_folder = data_folder / "Images"
    save_path = data_folder / "features.pt"

    img_names = [p.name for p in img_folder.iterdir() if p.is_file()]

    features_dict = {}
    for i in tqdm.tqdm(range(0, len(img_names), batch_size)):
        batch_names = img_names[i:i+batch_size]
        batch_dict = calc_and_save(
            model=model,
            img_folder=str(img_folder),
            img_names=batch_names,
            processor=processor,
            device=device
        )
        features_dict.update(batch_dict)

    end = time.time()
    torch.save(features_dict, save_path)

    print(f"Saved features calculated by CLIP at {save_path}; calculations took {end-start:.2f}s")

if __name__ == "__main__":

    MODEL_NAME = "openai/clip-vit-base-patch32"
    BATCH_SIZE = 512
    main(model_name=MODEL_NAME, batch_size=BATCH_SIZE)
