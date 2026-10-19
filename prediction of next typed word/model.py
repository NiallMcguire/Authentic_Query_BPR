"""Causal EEG-only next-word models sharing one input/output contract."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


ARCHITECTURES = ("mlp", "gru", "lstm", "tcn", "transformer")


class CausalTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(0.2)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        channels_first = states.transpose(1, 2)
        convolved = self.convolution(functional.pad(channels_first, (self.left_padding, 0)))
        updated = states + self.dropout(functional.gelu(convolved.transpose(1, 2)))
        return self.normalization(updated)


class EEGNextWord(nn.Module):
    def __init__(self, feature_dim: int, vocabulary_size: int, hidden_dim: int = 128,
                 architecture: str = "gru", num_layers: int = 1,
                 attention_heads: int = 4, kernel_size: int = 3):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if min(feature_dim, vocabulary_size, hidden_dim, num_layers, attention_heads, kernel_size) < 1:
            raise ValueError("Model dimensions, layers, heads, and kernel size must be positive")
        if architecture == "transformer" and hidden_dim % attention_heads:
            raise ValueError("Transformer hidden_dim must be divisible by attention_heads")
        self.architecture = architecture
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                     nn.LayerNorm(hidden_dim), nn.Dropout(0.2))
        if architecture in {"gru", "lstm"}:
            recurrent_class = nn.GRU if architecture == "gru" else nn.LSTM
            self.recurrent = recurrent_class(hidden_dim, hidden_dim, num_layers=num_layers,
                                              dropout=0.2 if num_layers > 1 else 0.0, batch_first=True)
        elif architecture == "tcn":
            self.temporal = nn.Sequential(*(CausalTemporalBlock(hidden_dim, kernel_size, 2 ** layer)
                                             for layer in range(num_layers)))
        elif architecture == "transformer":
            layer = nn.TransformerEncoderLayer(hidden_dim, attention_heads, hidden_dim * 4,
                                                dropout=0.2, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, eeg: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(eeg)
        if self.architecture in {"gru", "lstm"}:
            states, hidden = self.recurrent(encoded)
        elif self.architecture == "tcn":
            states = self.temporal(encoded)
        elif self.architecture == "transformer":
            steps, hidden_dim = encoded.shape[1:]
            positions = torch.arange(steps, device=eeg.device, dtype=encoded.dtype).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, hidden_dim, 2, device=eeg.device,
                                                 dtype=encoded.dtype) * (-math.log(10000.0) / hidden_dim))
            positional = torch.zeros(steps, hidden_dim, device=eeg.device, dtype=encoded.dtype)
            positional[:, 0::2] = torch.sin(positions * frequencies)
            positional[:, 1::2] = torch.cos(positions * frequencies[:hidden_dim // 2])
            causal_mask = torch.ones(steps, steps, dtype=torch.bool, device=eeg.device).triu(1)
            states = self.temporal(encoded + positional, mask=causal_mask,
                                   src_key_padding_mask=padding_mask, is_causal=True)
        else:
            states = encoded
        logits = self.head(states)
        return logits.masked_fill(padding_mask.unsqueeze(-1), 0.0) if padding_mask is not None else logits