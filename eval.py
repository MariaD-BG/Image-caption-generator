import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore, METEORScore

from src.dataset import ImageDataset, Vocabulary, collate_fn
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

            loss = criterion(output.permute(0, 2, 1)[:,:,1:], captions[:, 1:])
            total_loss += loss.item()

    return total_loss / len(data_loader)

def bleu_test(model: ImageCaptionModel, data_loader: DataLoader, vocab: Vocabulary, device: torch.device):
    model.eval()

    total_bleu = 0
    total_meteor = 0

    with torch.no_grad():
        for features, captions in data_loader:
            features, captions = features.to(device), captions.to(device)
            gen = model.generate(features, vocab)
            caps_list = [captions[i].tolist() for i in range(captions.shape[0])]
            caps_str = [ [vocab.itos[x] for x in res] for res in caps_list ]
            caps_str = [sent[:sent.index("<EOS>")] if "<EOS>" in sent else sent for sent in caps_str]
            caps_str = [s[1:] for s in caps_str]

            generated = [" ".join(x) for x in gen]
            caption = [" ".join(x) for x in caps_str]

            gt = [[c] for c in caption]

            total_bleu += bleu(generated, gt)
            total_meteor += meteor(generated, gt)

    return total_bleu / len(data_loader), total_meteor/len(data_loader)



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

    loss = evaluate(icg_model, test_loader, criterion, device)
    print(f"Test loss: {loss}")
    bleu_s, meteor_s = bleu_test(icg_model, test_loader, vocab, device)
    print(f"Test bleu: {bleu_s}")
    print(f"Test meteor: {meteor_s}")