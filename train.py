import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src import ImageCaptionModel, ImageDataset, Vocabulary
from src.dataset import collate_fn
from src.utils import plot_loss, save_checkpoint

CONFIG_PATH = "src/config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

input_dim = config["clip"]["input_dim"]
freq_thredhold = config["data"]["words_freq_threshold"]
data_path = config["data"]["folder_path"]
captions_path = data_path + "captions.txt"
features_path = data_path + "features.pt"

embed_size = config["model_params"]["embed_size"]
hidden_size = config["model_params"]["hidden_size"]
batch_size = config["training"]["batch_size"]
lr = float(config["training"]["lr"])
weight_decay = float(config["training"]["weight_decay"])
dropout = float(config["training"]["dropout"])
num_epochs = config["training"]["epochs"]

plots_path = config["saving"]["plots_path"]
checkpoints_path = config["saving"]["checkpoints_path"]

vocab = Vocabulary(freq_threshold = freq_thredhold)
vocab.build_vocabulary(captions_path = captions_path)
vocab_size = len(vocab)
train_dataset = ImageDataset(features_path = features_path, cap_path = captions_path, vocab = vocab, split = "train")
val_dataset = ImageDataset(features_path = features_path, cap_path = captions_path, vocab = vocab, split = "validation")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(input_dim=input_dim, embed_size = embed_size, hidden_size = hidden_size, vocab_size = vocab_size, dropout = dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) # The padding index 0 will be ignored
optimizer = torch.optim.Adam(params=model.parameters(), lr = lr, weight_decay = weight_decay)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = True)

print("Starting training...")

train_losses = []
val_losses= []

best_val_loss = float("inf")
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

    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, avg loss: {avg_loss}")

    if (epoch+1)%20 == 0:
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
            save_checkpoint(model=model, optimizer=optimizer, filepath = checkpoints_path+"checkpoint.pth")
            best_val_loss = val_loss

    if (epoch+1)%100 == 0:
        plot_loss(train_losses, val_losses, plots_path + "train_loss.png")