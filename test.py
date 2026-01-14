from dataset import Vocabulary, ImageDataset, collate_fn
from torch.utils.data import DataLoader

vocab = Vocabulary(freq_threshold = 5)
vocab.build_vocabulary(captions_path = "data/captions.txt")

dataset = ImageDataset(img_path = "data/Images", cap_path = "data/captions.txt", vocab = vocab)

print(f"len: {len(dataset)}")
print(f"datapoint: {dataset[42]}")
print(f"img tensor shape: {dataset[42][0].shape}")

loader = DataLoader(dataset = dataset, batch_size=32, collate_fn=collate_fn)