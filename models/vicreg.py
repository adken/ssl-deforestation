"""VICReg dual-encoder network for paired Sentinel-1/Sentinel-2 time series."""

import torch
import torch.nn as nn

from models.tempCNN import TempCNN


class _Expander(nn.Sequential):
    """Three-layer projection MLP used only during VICReg pretraining."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


class VICRegNet(nn.Module):
    """Separate TempCNN encoders and expanders for S1 SAR and S2 optical data."""

    def __init__(self, hidden_dim: int = 128, expander_dim: int = 256) -> None:
        super().__init__()
        self.encoder_s1 = TempCNN(
            input_dim=2, kernel_size=7, hidden_dims=hidden_dim, dropout=0.5
        )
        self.encoder_s2 = TempCNN(
            input_dim=10, kernel_size=7, hidden_dims=hidden_dim, dropout=0.5
        )
        self.expander_s1 = _Expander(hidden_dim, expander_dim)
        self.expander_s2 = _Expander(hidden_dim, expander_dim)

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repr_s1 = self.encoder_s1(s1)
        repr_s2 = self.encoder_s2(s2)
        return self.expander_s1(repr_s1), self.expander_s2(repr_s2)
