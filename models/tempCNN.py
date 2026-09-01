"""Shared TempCNN backbone used by both SSL pretraining and downstream tasks.

PyTorch re-implementation of Pelletier et al. (2019):
https://github.com/charlotte-pel/temporalCNN

Input convention: ``(N, C, T)`` (channels first, as expected by ``nn.Conv1d``).
Output: ``(N, hidden_dims)`` after global average pooling.

The historical layer attribute names (``conv_bn_relu1`` ... ``conv_bn_relu3``)
are intentionally preserved so thesis-era VICReg checkpoints remain loadable.
"""

import os

import torch
import torch.nn as nn


class Conv1D_BatchNorm_Relu_Dropout(nn.Module):
    """Single Conv1D -> BatchNorm -> ReLU -> Dropout block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        kernel_size: int = 5,
        drop_probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_dim,
                hidden_dims,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TempCNN(nn.Module):
    """Three-layer temporal CNN encoder with global average pooling."""

    def __init__(
        self,
        input_dim: int = 10,
        kernel_size: int = 7,
        hidden_dims: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.modelname = (
            f"TempCNN_input-dim={input_dim}_kernelsize={kernel_size}"
            f"_hidden-dims={hidden_dims}_dropout={dropout}"
        )

        # Keep historical names for compatibility with original checkpoints.
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
            input_dim, hidden_dims, kernel_size, dropout
        )
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size, dropout
        )
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size, dropout
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` with shape ``(N, C, T)`` into ``(N, hidden_dims)``."""
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.pool(x)
        return x.squeeze(-1)

    def save(self, path: str = "model.pth", **kwargs) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(dict(model_state=self.state_dict(), **kwargs), path)

    def load(self, path: str) -> dict:
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop("model_state", snapshot)
        self.load_state_dict(model_state)
        return snapshot
