"""Baseline sign classifier: a bidirectional GRU over landmark sequences.

This is the "before" half of the comparison. It is deliberately plain — no attention,
no positional encoding, no regularisation beyond a single dropout — so that whatever
the Transformer gains is attributable to the architecture rather than to extra tuning.
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineGRU(nn.Module):
    """Two-layer bidirectional GRU with masked mean-pooling and a linear head.

    Padding is handled two ways at once, because for a *bidirectional* RNN masking the
    pooling alone is not enough. Masked pooling keeps padded steps out of the classifier,
    but the backward direction would still start at the end of the padding and carry it
    into the real frames' hidden states. `pack_padded_sequence` fixes that properly by
    running each direction over each sequence's true length, and with
    `enforce_sorted=False` it needs no manual sorting — PyTorch permutes internally. So
    the sequence is packed for the recurrence and masked again for the pooling, and the
    result is genuinely independent of whatever sits in the padded positions (which
    test_models.py asserts).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Build the baseline.

        Args:
            num_features: Width of the per-frame feature vector.
            num_classes: Number of sign classes.
            hidden_dim: Hidden size per direction.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout applied between GRU layers and before the head.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional, so the pooled representation is 2 * hidden_dim wide.
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, landmarks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Classify a batch of padded sequences.

        Args:
            landmarks: Float tensor of shape (batch, seq_len, num_features).
            mask: Bool tensor of shape (batch, seq_len); True marks real frames.

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Lengths must live on the CPU for packing, regardless of the model's device.
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            landmarks, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=landmarks.size(1)
        )  # (batch, seq_len, 2 * hidden_dim)

        # Zero the padded steps, sum, then divide by the real length: a mean over real
        # frames only. clamp(min=1) guards the degenerate all-padding row.
        weights = mask.unsqueeze(-1).to(outputs.dtype)
        summed = (outputs * weights).sum(dim=1)
        lengths = weights.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths

        return self.classifier(self.dropout(pooled))
