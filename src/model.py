import torch
import torch.nn as nn
import numpy as np
from typing import List
from transformers import CLIPTokenizer

class ImageCaptionModel(nn.Module):
    def __init__(self, input_dim:int, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int = 1, dropout:float = 0.0):

        super(ImageCaptionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear_img = nn.Linear(input_dim, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.vocab_size = self.tokenizer.vocab_size


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

    def generate(self, features: torch.tensor, max_len: int = 20, beam_width: int = 5) -> List[str]:
        return self.beam_search(features, max_len, beam_width)

    def beam_search(self, features: torch.tensor, max_len: int = 20, beam_width: int = 5) -> List[str]:
        self.eval()
        batch_size = features.shape[0]

        with torch.no_grad():
            
            image_embeds = self.linear_img(features).unsqueeze(1)
            _, initial_states = self.lstm(image_embeds) # initial_states = (h_n, c_n) both of shape (1, batch_size, hidden_size)

            # Start with the <SOS> token
            start_token = self.tokenizer.bos_token_id
            
            # Initialize beams
            beams = []
            for i in range(batch_size):
                # Extract the initial hidden and cell states for this specific image 'i'
                h_i = initial_states[0][:, i:i+1, :] # shape (1, 1, hidden_size)
                c_i = initial_states[1][:, i:i+1, :] # shape (1, 1, hidden_size)
                initial_state_for_image = (h_i, c_i)
                beams.append([(torch.tensor([start_token]).to(features.device), 0.0, initial_state_for_image)])
            
            final_captions = [[] for _ in range(batch_size)]

            for _ in range(max_len):
                for i in range(batch_size):
                    if not beams[i]:
                        continue

                    new_candidates = []
                    for sequence, score, current_states in beams[i]: # Use current_states for the beam
                        if sequence[-1] == self.tokenizer.eos_token_id:
                            final_captions[i].append((sequence, score))
                            continue
                        
                        inputs = self.embed(sequence[-1].unsqueeze(0).unsqueeze(0))
                        hiddens, new_states = self.lstm(inputs, current_states)
                        output = self.linear_out(hiddens.squeeze(1))
                        
                        # Use log_softmax for numerical stability
                        log_probs = torch.nn.functional.log_softmax(output, dim=1)
                        top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=1)

                        for k in range(beam_width):
                            new_seq = torch.cat([sequence, top_k_indices[0, k].unsqueeze(0)])
                            new_score = score + top_k_log_probs[0, k].item()
                            new_candidates.append((new_seq, new_score, new_states))
                    
                    # Sort candidates by score and select top beam_width
                    beams[i] = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            for i in range(batch_size):
                # If no complete caption was found, use the best available beam
                if not final_captions[i]:
                    final_captions[i] = [(beams[i][0][0], beams[i][0][1])] if beams[i] else []


            # Decode the final captions
            decoded_captions = []
            for i in range(batch_size):
                if final_captions[i]:
                    best_seq, _ = max(final_captions[i], key=lambda x: x[1])
                    decoded_captions.append(self.tokenizer.decode(best_seq, skip_special_tokens=True))
                else:
                    decoded_captions.append("") # Or some default caption

        return decoded_captions