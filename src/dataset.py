import torch
import tqdm
import re
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPTokenizer

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, features_path: str, cap_path: str, split: str):

        if split not in ["train", "test", "validation"]:
            raise ValueError("Invalid split argument")

        self.cap_path = cap_path
        self.features_path = features_path

        self.features_dict = torch.load(features_path)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.data = []
        available_imgs = set(self.features_dict.keys())
        with open(cap_path, "r") as f:
            for line in tqdm.tqdm(f):
                img_name, cap = line.split(",", maxsplit=1)
                # Remove punctuation, convert to lowercase, and strip whitespace
                cap = re.sub(r'[^\w\s]', '', cap).lower().strip()
                if img_name not in available_imgs:
                    continue
                self.data.append((self.features_dict[img_name], cap))

        train_end = int(0.8*len(self.data))
        test_end = int(0.9*len(self.data))
        data_train = self.data[:train_end]
        data_test = self.data[train_end:test_end]
        data_valid = self.data[test_end:]

        if split == "train":
            self.data = data_train
        if split == "test":
            self.data = data_test
        if split == "validation":
            self.data = data_valid

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[str, torch.Tensor]:
        img, cap = self.data[index]
        
        # Tokenize the caption
        cap_tokens = self.tokenizer(cap, padding="max_length", max_length=77, truncation=True, return_tensors="pt")["input_ids"].squeeze(0)

        return (img, cap_tokens)

def collate_fn(batch):
    images, captions = zip(*batch)
    images_tensor = torch.stack(images)
    captions_tensor = torch.stack(captions)
    return images_tensor, captions_tensor
