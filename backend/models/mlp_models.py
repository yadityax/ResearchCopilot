"""
Lightweight MLP projection layers on top of pretrained sentence-transformer embeddings.
These are NOT trained from scratch — they are initialized with sensible defaults
(identity-like) and can be fine-tuned if needed.

Architecture from poster:
  sentence-transformers (pretrained, 384-dim)
      → EncoderMLP   (2-layer MLP, 384→384)
      → SinusoidalMLP (sinusoidal positional encoding layer, 384→384)
      → ChromaDB similarity search
"""
import math
import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    """
    2-layer MLP that projects sentence-transformer embeddings.
    Initialized close to identity so it doesn't distort pretrained embeddings.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 384, output_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection to preserve pretrained embedding quality
        return x + self.net(x)


class SinusoidalMLP(nn.Module):
    """
    Applies sinusoidal positional encoding to enrich embedding dimensions,
    then passes through a linear layer.
    Inspired by transformer positional encodings applied in embedding space.
    """

    def __init__(self, dim: int = 384):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)  # start as identity
        nn.init.zeros_(self.proj.bias)

        # Precompute sinusoidal encoding table
        pe = torch.zeros(1, dim)
        position = torch.arange(0, dim, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[0, 0::2] = torch.sin(position[0, 0::2] * div_term)
        if dim % 2 == 0:
            pe[0, 1::2] = torch.cos(position[0, 1::2] * div_term)
        else:
            pe[0, 1::2] = torch.cos(position[0, 1::2] * div_term[: dim // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sinusoidal encoding (broadcast over batch)
        x = x + 0.01 * self.pe.to(x.device)
        return self.proj(x)
