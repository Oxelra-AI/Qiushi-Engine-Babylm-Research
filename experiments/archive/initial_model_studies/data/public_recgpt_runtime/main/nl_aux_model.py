"""Auxiliary model for NextLat latent prediction"""
import torch
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass

@dataclass
class NL_Aux_Config:
    input_hidden_size: int # Input size (Concatenation of hidden and next token embedding vectors)
    hidden_size: int = -1 # Residual dimension (Projected down from concatenated input), if set to -1 defaults to input_hidden_size
    intermediate_size: int = 2560 # MLP intermediate dimension

class NL_Aux_Model(nn.Module):
    # Takes in the normalized hidden states of the main model along with the embeddings of the next token, and predicts the hidden states of the next token.
    def __init__(self, config: NL_Aux_Config):
        super().__init__()
        self.config = config

        if config.hidden_size == -1:
            config.hidden_size = config.input_hidden_size

        self.input_proj = nn.Linear(config.input_hidden_size * 2, config.hidden_size, bias=False)
        if config.input_hidden_size != config.hidden_size: # Allow for aux models hidden size to be different from main model. This could be useful to create a bottleneck.
            self.out_proj = nn.Linear(config.hidden_size, config.input_hidden_size, bias=False)
        else:
            self.out_proj = nn.Identity()
        
        # 3-Layer MLP
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mid = nn.Linear(config.intermediate_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        nn.init.zeros_(self.down.weight)
        nn.init.ones_(self.mid.weight)

        self.final_norm = nn.Parameter(torch.ones(config.input_hidden_size))

    def predict_next(self, hidden, embeddings):
        x = torch.cat([hidden, embeddings.detach()], dim=-1) # Concatenate hidden with next token's embedding
        x = self.input_proj(x)

        # 3-Layer MLP with GeLU activations
        x = self.up(x)
        x = F.gelu(x)
        x = self.mid(x)
        x = F.gelu(x)
        x = self.down(x)

        x = F.rms_norm(self.out_proj(x) + hidden, (x.size(-1),), self.final_norm, 1e-6) # Residual connection
        # Note: The residual makes it equivalent to predicting the diff between the next and current hidden states.
        return x

    def forward(self, hidden, embeddings, segment_ids, depth: int = 1): 
        # Inputs expected in shape [batch, seq_len, hidden_size], except segment IDs which are [batch, seq_len].
        # Returns loss only
        # We use detach for embeddings and targets to ensure gradients only flow backwards through the input hiddens.
        assert depth >= 1

        current = hidden
        valid = None
        losses = []

        for k in range(1, depth + 1):
            if hidden.size(1) <= k:
                break

            x = self.predict_next(current[:, :-1], embeddings[:, k:])

            # Prevent cross-document and padding token prediction across the full rollout path.
            valid_transition = (segment_ids[:, k - 1:-1] >= 0) & (segment_ids[:, k - 1:-1] == segment_ids[:, k:])
            valid = valid_transition if valid is None else valid[:, :-1] & valid_transition

            pred = x[valid]
            if pred.numel() > 0:
                target = hidden[:, k:][valid].detach()
                losses.append(F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1).mean()) # Sum over hidden dim

            current = x

        if not losses:  # Return zero loss if no valid transitions
            return hidden.sum() * 0.0

        return torch.stack(losses).mean()
