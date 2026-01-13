import torch
import torch.nn as nn
import numpy as np

class ImageCaptionModel(nn.Module):
    def __init__(self, input_dim:int, embed_size:int, hidden_size:int, vocab_size:int, num_layers : int =1):

        super(ImageCaptionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear_img = nn.Linear_(input_dim, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features : np.ndarray, captions: np.ndarray):
        """

            # features shape: (batch_size, input_dim)
            # captions shape: (batch_size, sequence_length)
        """
        embeddings = self.embed(captions[:, :-1]) # remove the <end> token

        # squash the features from the encoder to fit our desiredes embedding dimension
        image_embed = self.linear_img(features)
        image_embed = image_embed.unsqueeze(1)

        # Concatenate the image embedding first; like the image being the first 'word' we start predicting after
        inputs = torch.cat((image_embed, embeddings), dim=1)

        # Run the LSTM
        lstm_out, _ = self.lstm(inputs)

        # Predict Next Words
        outputs = self.linear_out(lstm_out)
        return outputs