import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import ImageCaptionModel
from .dataset import Vocabulary, collate_fn
from .dataset import ImageDataset

def evaluate(model : ImageCaptionModel, data_loader : DataLoader, criterion : nn.Module, device : torch.device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, captions in data_loader:
            features, captions = features.to(device), captions.to(device)
            output = model(features, captions)

            loss = criterion(output.permute(0, 2, 1)[:,:,1:], captions[:, 1:])
            total_loss += loss.item()

    return total_loss / len(data_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768

    vocab = Vocabulary(freq_threshold = 10)
    vocab.build_vocabulary(captions_path = "data/captions.txt")
    vocab_size = len(vocab)

    icg_model = ImageCaptionModel(input_dim=input_dim, embed_size = 128, hidden_size=32, vocab_size = vocab_size, dropout = 0.2).to(device)
    checkpoint = torch.load(args.checkpoint)
    icg_model.load_state_dict(checkpoint["model_state_dict"])

    eval_dataset = ImageDataset(features_path = "data/features.pt", cap_path = "data/captions.txt", vocab = vocab, split = "test")
    test_loader = DataLoader(dataset = eval_dataset, batch_size=256, collate_fn=collate_fn, shuffle = True)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # The padding index 0 will be ignored

    evaluate(icg_model, test_loader, criterion, device)

