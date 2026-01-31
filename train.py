import time
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from src import ImageCaptionModel, ImageDataset
from src.dataset import collate_fn
from src.utils import plot_loss

def save_checkpoint(model : ImageCaptionModel, optimizer=None, filepath : str ="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

CONFIG_PATH = "src/config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

input_dim = config["data"]["input_dim"]
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

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vocab_size = tokenizer.vocab_size

train_dataset = ImageDataset(features_path = features_path, cap_path = captions_path, split = "train")
val_dataset = ImageDataset(features_path = features_path, cap_path = captions_path, split = "validation")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(input_dim=input_dim, embed_size = embed_size, hidden_size = hidden_size, vocab_size = vocab_size, dropout = dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(params=model.parameters(), lr = lr, weight_decay = weight_decay)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle = True)

print("Starting training...")

train_losses = []
val_losses= []

best_val_loss = float("inf")
for epoch in range(num_epochs):

    print(f"Starting epoch...{epoch}")

    epoch_start = time.time()

    model.train()
    avg_loss = 0
    for batch in train_loader:
        features, captions = batch
        features, captions = features.to(device), captions.to(device)

        outputs = model(features, captions)
        loss = criterion(outputs.reshape(-1, vocab_size), captions.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        avg_loss += loss.item()
    avg_loss = avg_loss / len(train_loader)

    train_losses.append(avg_loss)

    epoch_end = time.time()
    print(f"Completed epoch {epoch} in {epoch_end - epoch_start}s")

    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss}")

    if (epoch+1)%20 == 0:
        model.eval()
        val_loss = 0
        for batch in val_loader:
            features, captions = batch
            features, captions = features.to(device), captions.to(device)
            with torch.no_grad():
                outputs = model(features, captions)
                loss = criterion(outputs.reshape(-1, vocab_size), captions.reshape(-1))

            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        print(f"Calculated validation loss at epoch {epoch+1}: {val_loss}")
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            save_checkpoint(model=model, optimizer=optimizer, filepath = checkpoints_path+"checkpoint.pth")
            best_val_loss = val_loss

    if (epoch+1)%100 == 0:
        plot_loss(train_losses, val_losses, plots_path + "train_loss.png")
