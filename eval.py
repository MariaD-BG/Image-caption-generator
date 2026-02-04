"""
Script for model evaluation;
Calculates loss and BLEU score on test set
"""

import argparse
import yaml
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from transformers import CLIPTokenizer

from src.dataset import ImageDataset, collate_fn
from src.model import ImageCaptionModel, ModelConfig

BLEU = BLEUScore(n_gram=4)

def evaluate(
        model : ImageCaptionModel,
        data_loader : DataLoader,
        criterion : nn.Module,
        device : torch.device
    ) -> float:

    """
    Calculates loss on test set
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, captions in data_loader:
            features, captions = features.to(device), captions.to(device)
            output = model(features, captions)

            loss = criterion(output.reshape(-1, model.config.vocab_size), captions.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)

def bleu_test(
        model: ImageCaptionModel,
        data_loader: DataLoader,
        tokenizer: CLIPTokenizer,
        device: torch.device
    ) -> None:

    """
    Calculates BLEU score on test set
    """

    model.eval()

    total_bleu = 0

    with torch.no_grad():
        for features, captions in tqdm.tqdm(data_loader):
            features = features.to(device)
            print(f"Features shape: {features.shape}")
            gen = model.generate(features)

            caps_str = tokenizer.batch_decode(captions, skip_special_tokens=True)

            gt = [[c] for c in caps_str]

            total_bleu += BLEU(gen, gt)

    return total_bleu / len(data_loader)

def main(args:argparse.Namespace, config_path:str) -> None:

    """main method for evaluation"""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    vocab_size = tokenizer.vocab_size

    model_config = ModelConfig(
        input_dim=config["data"]["input_dim"],
        embed_size=config["model_params"]["embed_size"],
        hidden_size=config["model_params"]["hidden_size"],
        vocab_size=vocab_size,
        num_layers=config["model_params"]["lstm_layers"],
        dropout=config["training"]["dropout"]
    )

    icg_model = ImageCaptionModel(model_config).to(device)
    checkpoint = torch.load(args.checkpoint)
    icg_model.load_state_dict(checkpoint["model_state_dict"])

    eval_dataset = ImageDataset(
        features_path = "data/features.pt",
        cap_path = "data/captions.txt",
        split = "test"
    )
    test_loader = DataLoader(
        dataset = eval_dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=collate_fn,
        shuffle = True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    loss = evaluate(icg_model, test_loader, criterion, device)
    print(f"Test loss: {loss}")
    bleu_s = bleu_test(icg_model, test_loader, tokenizer, device)
    print(f"Test BLEU: {bleu_s}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args_parsed = parser.parse_args()

    CONFIG_PATH = "src/config.yaml"

    main(args=args_parsed, config_path=CONFIG_PATH)
