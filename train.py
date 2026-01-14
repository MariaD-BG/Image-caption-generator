import torch
import torch.nn as nn
import tqdm
from decoder.model import ImageCaptionModel
from torch.utils.data import Dataset, DataLoader
from dataset import Vocabulary, ImageDataset, collate_fn
from encoder.clip import extract_clip_features

input_dim = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(input_dim=input_dim, embed_size = 256, hidden_size=64, vocab_size = 5000).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # The padding index 0 will be ignored
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

vocab = Vocabulary(freq_threshold = 5)
vocab.build_vocabulary(captions_path = "data/captions.txt")
dataset = ImageDataset(img_path = "data/Images", cap_path = "data/captions.txt", vocab = vocab)

loader = DataLoader(dataset = dataset, batch_size=32, collate_fn=collate_fn)

print("Starting training...")

num_epochs = 100
for epoch in range(num_epochs):
    avg_loss = 0
    print(f"Starting epoch {epoch}")
    for batch in tqdm.tqdm(loader):
        images, captions = batch
        captions = captions.to(device)
        features = extract_clip_features(images).to(device)
        output = model(features, captions)
        loss = criterion(output.permute(0, 2, 1)[:,:,1:], captions[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss = avg_loss / len(loader)

    if epoch%10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, avg loss: {avg_loss}")