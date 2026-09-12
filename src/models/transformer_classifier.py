"""Improved sign classifier: a small Transformer encoder with attention pooling.

Three deliberate changes over the GRU baseline, each with a reason specific to this task:

1. **Self-attention** lets any frame consult any other directly. A sign's meaning often
   depends on the relation between distant moments — where the hand starts versus where
   it ends — which a recurrent state has to carry step by step.
2. **Learned positional embeddings**, because self-attention alone is order-invariant and
   these sequences are fundamentally temporal. They are learned rather than sinusoidal
   since `max_len` is small and fixed, so there is nothing to extrapolate to.
3. **Attention pooling** replaces mean-pooling. A mean treats the frames where the hands
   are still and the frames carrying the actual handshape as equally informative;
   a learned query lets the model decide which frames the decision rests on — and its
   weights are inspectable, which makes the improvement explainable rather than magical.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Pool a sequence into one vector using a single learned query.

    Equivalent to one attention head whose query is a parameter rather than a function
    of the input: it scores every frame and returns their weighted average.
    """

    def __init__(self, embed_dim: int) -> None:
        """Build the pooling layer.

        Args:
            embed_dim: Width of the encoder output.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim) * embed_dim**-0.5)
        self.scale = embed_dim**-0.5

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reduce a padded sequence to a single vector.

        Args:
            sequence: Float tensor of shape (batch, seq_len, embed_dim).
            mask: Bool tensor of shape (batch, seq_len); True marks real frames.

        Returns:
            Pooled tensor of shape (batch, embed_dim).
        """
        scores = (sequence @ self.query) * self.scale  # (batch, seq_len)
        # -inf on padded positions makes their softmax weight exactly zero.
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bs,bsd->bd", weights, sequence)


class TransformerClassifier(nn.Module):
    """Transformer encoder over landmark frames, pooled by a learned query.

    Attributes:
        pool: The attention-pooling layer, exposed so its weights can be inspected.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        max_len: int,
        embed_dim: int = 192,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.3,
    ) -> None:
        """Build the model.

        Args:
            num_features: Width of the per-frame feature vector.
            num_classes: Number of sign classes.
            max_len: Sequence length the positional embedding is sized for.
            embed_dim: Encoder width. Must be divisible by `num_heads`.
            num_heads: Attention heads per layer.
            num_layers: Encoder layers.
            ff_dim: Hidden width of each layer's feed-forward block.
            dropout: Dropout used throughout.

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.max_len = max_len
        self.input_projection = nn.Linear(num_features, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: noticeably more stable for small models
        )
        # The nested-tensor fast path does not apply to pre-norm layers; saying so
        # explicitly keeps PyTorch from warning about it on every instantiation.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.pool = AttentionPooling(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, landmarks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Classify a batch of padded sequences.

        Args:
            landmarks: Float tensor of shape (batch, seq_len, num_features).
            mask: Bool tensor of shape (batch, seq_len); True marks real frames.

        Returns:
            Logits of shape (batch, num_classes).

        Raises:
            ValueError: If the input is longer than the positional embedding.
        """
        seq_len = landmarks.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        hidden = self.input_projection(landmarks) + self.positional_embedding[:, :seq_len]
        hidden = self.input_dropout(self.input_norm(hidden))

        # PyTorch's convention is the inverse of ours: src_key_padding_mask marks the
        # positions to IGNORE, so it takes ~mask. Getting this backwards is the classic
        # silent bug here - it trains, but attends to padding and ignores real frames -
        # so test_models.py asserts that padding cannot change a real frame's output.
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)

        return self.head(self.pool(hidden, mask))
