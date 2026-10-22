"""Trainable ALFRED-inspired (not ALFRED-equivalent) gated fusion head."""

import torch
from torch import nn


class GatedEmotionFusion(nn.Module):
    def __init__(
        self, text_dim: int, image_dim: int, hidden_dim: int = 128, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.text = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.image = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.emotion_head = nn.Linear(hidden_dim, 6)
        self.sentiment_head = nn.Linear(hidden_dim, 3)

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_hidden, image_hidden = self.text(text), self.image(image)
        gate = self.gate(torch.cat((text_hidden, image_hidden), dim=-1))
        fused = gate * text_hidden + (1 - gate) * image_hidden
        return self.emotion_head(fused), self.sentiment_head(fused)
