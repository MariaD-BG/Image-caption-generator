import torch
import torch.nn as nn
import numpy as np

class ImageCaptionModel(nn.Module):
    def __init__(self, input_dim:int, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int = 1, dropout:float = 0.0):

        super(ImageCaptionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear_img = nn.Linear(input_dim, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features : np.ndarray, captions: np.ndarray):
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

    def generate(self, features, vocab, max_len : int = 20):
        """
        Generates a descrption given the feature vector of an image

        :param self: Description
        :param features: feature vector for target image
        :param vocab:
        :param max_len: upper bound for number of tokens per description
        """
        result = []

        with torch.no_grad():
            inputs = self.linear_img(features).unsqueeze(1)

            states = None
            next_token = None

            while len(result)<max_len and next_token != vocab.stoi["<EOS>"]:
                h, c = self.lstm(inputs, c)
                output = self.linear_out(h.squeeze(1))
                predicted = output.argmax(1)
                inputs = self.embed(predicted).unsqueeze(1)

        return [vocab.itos[idx] for idx in result]