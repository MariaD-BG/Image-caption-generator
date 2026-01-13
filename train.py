import torch
from decoder.model import ImageCaptionModel
from torch.utils.data import Dataset, Dataloader

input_dim = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(input_dim=input_dim, embed_size = 256, hidden_size=64, vocab_size = 5000).to(device)

num_epochs = 100
for epoch in range(num_epochs):
