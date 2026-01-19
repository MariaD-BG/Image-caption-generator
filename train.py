import torch
import torch.nn as nn
import tqdm
from decoder.model import ImageCaptionModel
from torch.utils.data import Dataset, DataLoader
from dataset import Vocabulary, ImageDataset, collate_fn
from utils import plot_loss, save_checkpoint

input_dim = 768

vocab = Vocabulary(freq_threshold = 10)
vocab.build_vocabulary(captions_path = "data/captions.txt")
vocab_size = len(vocab)
train_dataset = ImageDataset(features_path = "data/features.pt", cap_path = "data/captions.txt", vocab = vocab, split = "train")
val_dataset = ImageDataset(features_path = "data/features.pt", cap_path = "data/captions.txt", vocab = vocab, split = "validation")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(input_dim=input_dim, embed_size = 128, hidden_size=32, vocab_size = vocab_size, dropout = 0.2).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # The padding index 0 will be ignored
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader = DataLoader(dataset = train_dataset, batch_size=256, collate_fn=collate_fn, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size=256, collate_fn=collate_fn, shuffle = True)

print("Starting training...")

train_losses = []
val_losses= []

num_epochs = 1000
best_val_loss = 1e9
for epoch in range(num_epochs):

    avg_loss = 0
    for batch in train_loader:
        features, captions = batch
        features, captions = features.to(device), captions.to(device)
        output = model(features, captions)
        loss = criterion(output.permute(0, 2, 1)[:,:,1:], captions[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss = avg_loss / len(train_loader)

    train_losses.append(avg_loss)

    if epoch%10 == 9:
        print(f"Epoch {epoch}/{num_epochs}, avg loss: {avg_loss}")

    if epoch%20 == 19:
        val_loss = 0
        for batch in val_loader:
            features, captions = batch
            features, captions = features.to(device), captions.to(device)
            with torch.no_grad():
                output = model(features, captions)
                loss = criterion(output.permute(0, 2, 1)[:,:,1:], captions[:, 1:])

            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        print(f"Calculated validation loss at epoch {epoch}: {val_loss}")
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            save_checkpoint(model=model, optimizer=optimizer, filepath = "checkpoint.pth")
            best_val_loss = val_loss

    if epoch%100 == 99:
        plot_loss(train_losses, val_losses, "plots/train_loss.png")