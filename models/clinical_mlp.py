"""Lightweight clinical feature MLP.

Architecture follows the specification: Linear -> ReLU -> Dropout -> Linear,
producing a compact clinical feature embedding that is fused with the image
embedding via concatenation.
"""

from __future__ import annotations

import torch.nn as nn


class ClinicalMLP(nn.Module):
    """Encodes raw clinical features into a low-dimensional embedding."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the clinical MLP.

        Args:
            in_dim: Raw clinical feature dimensionality.
            hidden_dim: Hidden layer width.
            embedding_dim: Output embedding dimensionality.
            dropout: Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    @property
    def feature_dim(self) -> int:
        """Output embedding dimensionality."""
        return self.net[-1].out_features

    def forward(self, x: object) -> object:
        """Project raw clinical features to an embedding.

        Args:
            x: Clinical feature tensor of shape ``(B, in_dim)``.

        Returns:
            Embedding of shape ``(B, embedding_dim)``.
        """
        return self.net(x)