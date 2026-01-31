import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore, METEORScore
from transformers import CLIPTokenizer

from src.dataset import ImageDataset, collate_fn
from src.model import ImageCaptionModel

meteor = METEORScore()
bleu = BLEUScore(n_gram=4)

def evaluate(model : ImageCaptionModel, data_loader : DataLoader, criterion : nn.Module, device : torch.device) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, captions in data_loader:
            features, captions = features.to(device), captions.to(device)
            output = model(features, captions)
            
            loss = criterion(output.reshape(-1, model.vocab_size), captions.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)

def bleu_test(model: ImageCaptionModel, data_loader: DataLoader, tokenizer: CLIPTokenizer, device: torch.device):
    model.eval()

    total_bleu = 0
    total_meteor = 0

    with torch.no_grad():
        for features, captions in data_loader:
            features = features.to(device)
            gen = model.generate(features)
            
            caps_str = tokenizer.batch_decode(captions, skip_special_tokens=True)

            gt = [[c] for c in caps_str]

            total_bleu += bleu(gen, gt)
            total_meteor += meteor(gen, gt)

    return total_bleu / len(data_loader), total_meteor/len(data_loader)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 768

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size

    # These sizes should match the saved model's config
    embed_size = 512
    hidden_size = 512

    icg_model = ImageCaptionModel(input_dim=input_dim, embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size).to(device)
    checkpoint = torch.load(args.checkpoint)
    icg_model.load_state_dict(checkpoint["model_state_dict"])

    eval_dataset = ImageDataset(features_path = "data/features.pt", cap_path = "data/captions.txt", split = "test")
    test_loader = DataLoader(dataset = eval_dataset, batch_size=256, collate_fn=collate_fn, shuffle = True)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    loss = evaluate(icg_model, test_loader, criterion, device)
    print(f"Test loss: {loss}")
    bleu_s, meteor_s = bleu_test(icg_model, test_loader, tokenizer, device)
    print(f"Test bleu: {bleu_s}")
    print(f"Test meteor: {meteor_s}")
