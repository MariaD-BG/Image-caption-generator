import torch
import torch.nn as nn
import numpy as np
from typing import List

from src.dataset import Vocabulary

class ImageCaptionModel(nn.Module):
    def __init__(self, input_dim:int, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int = 1, dropout:float = 0.0):

        super(ImageCaptionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear_img = nn.Linear(input_dim, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features : np.ndarray, captions: np.ndarray) -> torch.tensor:
        """
            # features shape: (batch_size, input_dim)
            # captions shape: (batch_size, sequence_length)
        """
        embeddings = self.embed(captions[:, :-1]) # remove the <end> token

        # squash the features from the encoder to fit our desiredes embedding dimension
        image_embed = self.linear_img(features)
        image_embed = self.dropout(image_embed)
        image_embed = image_embed.unsqueeze(1)

        # Concatenate the image embedding first; like the image being the first 'word' we start predicting after
        inputs = torch.cat((image_embed, embeddings), dim=1)

        # Run the LSTM
        lstm_out, _ = self.lstm(inputs)

        # Predict Next Words
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear_out(lstm_out)
        return outputs

    def generate(self, features : torch.tensor, vocab: Vocabulary, max_len : int =20) -> List[str]:
        self.eval()
        batch_size = features.shape[0]
        result_tokens = [[] for _ in range(batch_size)]

        with torch.no_grad():

            inputs = self.linear_img(features).unsqueeze(1)

            # We run the LSTM once just to get the 'states' initialized with image info.
            # We DO NOT use the output of this step because we didn't train on it.
            _, states = self.lstm(inputs)

            # 2. Start the actual generation with the <SOS> token
            # We need to feed <SOS> to get the first real word (just like in training)
            start_token = vocab.stoi["<SOS>"]
            inputs = self.embed(torch.tensor([start_token] * batch_size).unsqueeze(1).to(features.device))

            for _ in range(max_len):
                # Pass inputs (token) and previous states (context from image + prev words)
                hiddens, states = self.lstm(inputs, states)

                # Predict next word
                output = self.linear_out(hiddens.squeeze(1))
                predicted = output.argmax(1)

                token_ids = predicted.tolist()

                result_tokens = [currlist + [elem] for currlist, elem in zip(result_tokens, token_ids)]

                # Stop if all lists have <EOS>
                if all([any([vocab.itos[token_id] == "<EOS>" for token_id in res]) for res in result_tokens]):
                    break

                # Prepare next input
                inputs = self.embed(predicted).unsqueeze(1)

        result_tokens = [[vocab.itos[idx] for idx in res] for res in result_tokens]
        final_captions = [sent[:sent.index("<EOS>")] if "<EOS>" in sent else sent for sent in result_tokens]

        return final_captions

