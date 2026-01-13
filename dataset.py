from utils import strip_syntax
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold:int) -> None:
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, captions_path:str):
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

