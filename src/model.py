"""
Definition of the Image Caption Generation model.
"""

from typing import List,Tuple
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from transformers import CLIPTokenizer

@dataclass
class ModelConfig:
    """Configuration for ImageCaptionModel."""
    input_dim: int
    embed_size: int
    hidden_size: int
    vocab_size: int
    num_layers: int = 1
    dropout: float = 0.0

class ImageCaptionModel(nn.Module):

    """
    Definition of Image Caption Generation model.
    Includes forward pass and generation & inference methods
    """

    def __init__(self,
                 config: ModelConfig
        ):

        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.linear_img = nn.Linear(config.input_dim, config.embed_size)
        self.lstm = nn.LSTM(
            config.embed_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True
        )
        self.linear_out = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = torch.nn.Dropout(config.dropout)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, features : np.ndarray, captions: np.ndarray) -> torch.Tensor:
        """
            # features shape: (batch_size, input_dim)
            # captions shape: (batch_size, sequence_length)
        """
        embeddings = self.embed(captions[:, :-1]) # remove the <end> token

        # squash the features from the encoder to fit our desiredes embedding dimension
        image_embed = self.linear_img(features)
        image_embed = self.dropout(image_embed)
        image_embed = image_embed.unsqueeze(1)

        # Concatenate the image embedding first;
        # like the image being the first 'word' we start predicting after
        inputs = torch.cat((image_embed, embeddings), dim=1)

        # Run the LSTM
        lstm_out, _ = self.lstm(inputs)

        # Predict Next Words
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear_out(lstm_out)
        return outputs

    def generate(self,
                 features: torch.Tensor,
                 max_len: int = 20,
                 beam_width: int = 5
        ) -> List[str]:
        """
        Call beam search on a batch of features to generate captions for images
        """
        return self.beam_search(features, max_len, beam_width)

    def beam_search(self,
                    features: torch.Tensor,
                    max_len: int = 20,
                    beam_width: int = 5
                ) -> List[str]:
        """
        Beam search for choosing the best caption based on accumulated score
        Better than greedy approach of always choosing the next best token
        """
        self.eval()

        with torch.no_grad():
            beams, _ = self._init_beam_search(features)
            batch_size = features.shape[0]
            final_captions:List[List[Tuple[torch.Tensor, float]]] = [[] for _ in range(batch_size)]

            print(f"Inferred batch size: {batch_size}")

            for _ in range(max_len):
                for i in range(batch_size):
                    if not beams[i]:
                        continue

                    # Extract the heavy logic into a helper method
                    beams[i], completed = self._expand_beam(
                        beams[i], beam_width
                    )
                    final_captions[i].extend(completed)

            return self._decode_captions(beams, final_captions, batch_size)

    def _init_beam_search(self, features: torch.Tensor):
        """Helper to initialize beams and states."""
        batch_size = features.shape[0]
        image_embeds = self.linear_img(features).unsqueeze(1)
        _, initial_states = self.lstm(image_embeds)
        start_token = self.tokenizer.bos_token_id

        beams = []
        for i in range(batch_size):
            h_i = initial_states[0][:, i:i+1, :]
            c_i = initial_states[1][:, i:i+1, :]
            # Tuple: (sequence, score, state)
            beams.append([(
                torch.tensor([start_token]).to(features.device),
                0.0,
                (h_i, c_i)
            )])
        return beams, initial_states

    def _expand_beam(self, current_beam, beam_width):

        """Helper to process one step of beam search for a single image."""
        new_candidates = []
        completed_captions = []

        for sequence, score, current_states in current_beam:
            if sequence[-1] == self.tokenizer.eos_token_id:
                completed_captions.append((sequence, score))
                continue

            inputs = self.embed(sequence[-1].view(1, 1))
            hiddens, new_states = self.lstm(inputs, current_states)
            output = self.linear_out(hiddens.squeeze(1))
            log_probs = torch.nn.functional.log_softmax(output, dim=1)
            top_k_log, top_k_idx = torch.topk(log_probs, beam_width, dim=1)

            for k in range(beam_width):
                new_seq = torch.cat([sequence, top_k_idx[0, k].unsqueeze(0)])
                new_score = score + top_k_log[0, k].item()
                new_candidates.append((new_seq, new_score, new_states))

        # Select top k
        best_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        return best_candidates, completed_captions

    def _decode_captions(self, beams, final_captions, batch_size) -> List[str]:
        """Helper to turn token IDs into strings."""
        decoded = []
        for i in range(batch_size):
            if not final_captions[i]:
                # Fallback to current beam if no EOS found
                final_captions[i] = [(beams[i][0][0], beams[i][0][1])] if beams[i] else []

            if final_captions[i]:
                best_seq, _ = max(final_captions[i], key=lambda x: x[1])
                decoded.append(self.tokenizer.decode(best_seq, skip_special_tokens=True))
            else:
                decoded.append("")
        return decoded
