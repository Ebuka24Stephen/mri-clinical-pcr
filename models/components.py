"""Reusable neural network building blocks."""

from __future__ import annotations

import torch.nn as nn


class MLPHead(nn.Module):
    """Multi-layer perceptron classification head."""

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        out_dim: int,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the MLP.

        Args:
            in_dim: Input feature dimensionality.
            hidden: Hidden layer widths.
            out_dim: Output dimensionality (number of classes).
            dropout: Dropout probability between linear layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    @property
    def feature_dim(self) -> int:
        """Input feature dimensionality."""
        return self.net[0].in_features

    def forward(self, x: object) -> object:
        """Return logits for a batch of features.

        Args:
            x: Feature tensor of shape ``(B, in_dim)``.

        Returns:
            Logits of shape ``(B, out_dim)``.
        """
        return self.net(x)