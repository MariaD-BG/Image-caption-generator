"""Utility functions used across the application."""

import re
from typing import List
import matplotlib.pyplot as plt

def plot_loss(train_loss: List[float],
              val_loss: List[float],
              save_path: str,
              val_interval: int = 20) -> None:

    """
    Docstring for plot_loss

    Plot the train and validation losses across the epochs

    :param train_loss: a list of floats that represent the respective train loss for each epoch
    :param val_loss: a list of floats that represent the validation loss for each epoch recorded
    :param save_path: a path to which the plot is to be saved
    :param val_interval: the validation loss is calculated every val_interval epochs
    """

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, marker='o', label='Training Loss', color='blue')
    val_indices = list(range(val_interval - 1, len(train_loss), val_interval))
    plt.plot(val_indices,
             val_loss,
             marker='s',
             linestyle='--',
             label='Validation Loss',
             color='red',
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss by Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def strip_syntax(txt:str) ->str:
    """
    Docstring for strip_syntax

    Strip the syntax from text

    :param txt: the string to be stripped
    :type txt: str
    :return: the string stripped
    :rtype: str
    """
    cleaned_txt = re.sub(r'[^a-zA-Z ]', '', txt)
    return cleaned_txt
