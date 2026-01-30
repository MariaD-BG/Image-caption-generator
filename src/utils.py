import re
import torch
import matplotlib.pyplot as plt
from typing import List

def plot_loss(train_loss: List[float], val_loss: List[float], save_path: str, val_interval: int = 20) -> None:

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, marker='o', label='Training Loss', color='blue')
    val_indices = [i for i in range(val_interval - 1, len(train_loss), val_interval)]
    plt.plot(val_indices, val_loss, marker='s', linestyle='--', label='Validation Loss', color='red')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss by Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def strip_syntax(txt:str) ->str:
    cleaned_txt = re.sub(r'[^a-zA-Z ]', '', txt)
    return cleaned_txt