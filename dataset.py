import torch
import tqdm
import numpy as np
from typing import Tuple, List
from utils import strip_syntax
from collections import Counter
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    """
    Class for handling words and additional tokens
    """
    def __init__(self, freq_threshold:int) -> None:
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, captions_path:str) -> None:
        frequencies = Counter()
        start_idx = 4

        with open(captions_path, 'r') as f:
            next(f) # skip first line
            for line in f:
                sentence = line.split(",", maxsplit=1)[1]
                sentence = strip_syntax(sentence.lower())
                words = sentence.split()
                frequencies.update(words)

        filtered_words = [x for x in frequencies if frequencies[x] >= self.freq_threshold]

        for id, word in enumerate(filtered_words):
            idx = id+start_idx
            self.stoi[word] = idx
            self.itos[idx] = word

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path: str, cap_path: str, vocab: Vocabulary):

        self.vocab = vocab
        self.img_path = img_path
        self.cap_path = cap_path

        img_folder = Path(img_path)
        available_imgs = set([img.name for img in img_folder.iterdir()])
        self.data = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),         # CLIP uses 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        print("Creating dataset...")
        with open(cap_path, "r") as f:
            for line in tqdm.tqdm(f):
                img_name, cap = line.split(",", maxsplit=1)
                cap = strip_syntax(cap.lower()).rstrip(' ')
                if img_name not in available_imgs:
                    continue
                self.data.append((img_name, cap))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[str, np.ndarray]:
        img, cap = self.data[index]
        cap = torch.tensor([self.vocab.stoi.get("<SOS>")] + [self.vocab.stoi.get(x,self.vocab.stoi["<UNK>"]) for x in cap.split(" ")] + [self.vocab.stoi.get("<EOS>")])
        img = Image.open(self.img_path + "/" + img).convert("RGB")
        img = self.transform(img)
        return (img, cap)

def collate_fn(batch : List[Tuple[str, str]]):
    images, captions = zip(*batch)
    images_tensor = torch.stack(images)
    captions_tensor = pad_sequence(captions, batch_first=True, padding_value  = 0)
    return images_tensor, captions_tensor