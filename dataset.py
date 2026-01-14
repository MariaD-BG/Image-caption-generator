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

    def __init__(self, features_path: str, cap_path: str, vocab: Vocabulary, split: str):

        if split not in ["train", "test", "validation"]:
            raise ValueError("Invalid split argument")

        self.vocab = vocab
        self.cap_path = cap_path
        self.features_path = features_path

        self.features_dict = torch.load(features_path)

        self.data = []
        available_imgs = set(self.features_dict.keys())
        with open(cap_path, "r") as f:
            for line in tqdm.tqdm(f):
                img_name, cap = line.split(",", maxsplit=1)
                cap = strip_syntax(cap.lower()).rstrip(' ')
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

    def __getitem__(self, index) -> Tuple[str, np.ndarray]:
        img, cap = self.data[index]
        cap = torch.tensor([self.vocab.stoi.get("<SOS>")] + [self.vocab.stoi.get(x,self.vocab.stoi["<UNK>"]) for x in cap.split(" ")] + [self.vocab.stoi.get("<EOS>")])
        return (img, cap)

def collate_fn(batch):
    images, captions = zip(*batch)
    images_tensor = torch.stack(images)
    captions_tensor = pad_sequence(captions, batch_first=True, padding_value  = 0)
    return images_tensor, captions_tensor