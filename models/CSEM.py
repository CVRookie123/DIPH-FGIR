import torch
import torch.nn as nn


class CSEM(nn.Module):
    """
    Channel-wise modulation module.

    This module computes a channel descriptor from the input feature map and
    uses it to generate a gating factor for feature recalibration.
    """

    def __init__(self, num_channels, epsilon=1e-5, mode="l2", after_relu=False):
        super().__init__()

        if mode not in {"l1", "l2"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'l1' or 'l2'.")

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W].

        Returns:
            torch.Tensor: Recalibrated feature map with the same shape as input.
        """
        if self.mode == "l2":
            embedding = (
                x.pow(2).sum((2, 3), keepdim=True) + self.epsilon
            ).pow(0.5) * self.alpha
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            abs_x = x if self.after_relu else torch.abs(x)
            embedding = abs_x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        gate = 1.0 + torch.tanh(embedding * norm + self.beta)
        return x * gate