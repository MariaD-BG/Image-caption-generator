import torch
import tqdm
import time
from pathlib import Path
from PIL import Image
from typing import List
from torchvision import transforms
from clip import extract_clip_features

transform = transforms.Compose([
    transforms.Resize((224, 224)),         # CLIP uses 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def calc_and_save(img_folder: str, img_names: List[str]) -> dict[str, torch.tensor]:

    tensors_list = []
    for img_name in img_names:
        img_path = Path(img_folder + "/" + img_name)
        image = Image.open(img_path).convert("RGB")
        tensors_list.append(transform(image))

    imgs = torch.stack(tensors_list)
    features = extract_clip_features(imgs)
    features_dict = {}
    for i, img_name in enumerate(img_names):
        features_dict[img_name] = features[i].cpu()
    return features_dict

if __name__ == "__main__":

    start = time.time()

    data_folder = Path(__file__).parent.parent / "data"
    img_folder = data_folder / "Images"
    save_path = str(data_folder) + "/features.pt"
    img_names = list(img_folder.iterdir())
    img_names = [str(img.name) for img in img_names]

    features_dict = {}
    batch_size = 512
    for i in tqdm.tqdm(range(0, len(img_names), batch_size)):
        j = min(len(img_names), i+batch_size)
        batch_dict = calc_and_save(str(img_folder), img_names[i:j])
        features_dict = {**features_dict, **batch_dict}

    end = time.time()
    torch.save(features_dict, save_path)

    print(f"Saved features calculated by CLIP at {save_path}; calculations took {end-start}s")
