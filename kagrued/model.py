"""Attention-based GRU encoder-decoder for multi-step workload forecasting (PyTorch).

This is a reference re-implementation of the model family described in the K-AGRUED paper: a GRU
encoder reads a window of past load and calendar features, and a GRU decoder emits the next
``horizon`` load values one step at a time. At every decoder step, additive (Bahdanau) attention
over all encoder states produces a context vector.

It was written from the paper's high-level description, not from the code that produced the
paper's numbers, so layer sizes and training settings are defaults chosen here.
"""

from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """score(h_dec, h_enc_t) = v^T tanh(W_dec h_dec + W_enc h_enc_t)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w_dec = nn.Linear(hidden, hidden, bias=False)
        self.w_enc = nn.Linear(hidden, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, dec_state: torch.Tensor, enc_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # dec_state: (B, H); enc_states: (B, T, H)
        scores = self.v(torch.tanh(self.w_dec(dec_state).unsqueeze(1) + self.w_enc(enc_states))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), enc_states).squeeze(1)  # (B, H)
        return context, weights


class AttentionGRUEncoderDecoder(nn.Module):
    """Input (B, lookback, n_features) -> output (B, horizon) of scaled load.

    Feature 0 of the input must be the (scaled) load: the decoder starts from the last observed
    value and feeds its own predictions back in. With ``residual=True`` each decoder step predicts
    the change from the previous value rather than the level itself.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.residual = residual
        self.encoder = nn.GRU(
            n_features, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = AdditiveAttention(hidden)
        self.decoder_cell = nn.GRUCell(1 + hidden, hidden)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        enc_states, h = self.encoder(x)  # (B, T, H), (L, B, H)
        state = h[-1]
        prev = x[:, -1, :1]  # last observed load, (B, 1)
        outputs, attn = [], []
        for _ in range(self.horizon):
            context, weights = self.attention(state, enc_states)
            state = self.decoder_cell(torch.cat([prev, context], dim=1), state)
            step = self.out(torch.cat([state, context], dim=1))
            prev = prev + step if self.residual else step
            outputs.append(prev)
            attn.append(weights)
        y = torch.cat(outputs, dim=1)  # (B, horizon)
        if return_attention:
            return y, torch.stack(attn, dim=1)  # (B, horizon, T)
        return y
