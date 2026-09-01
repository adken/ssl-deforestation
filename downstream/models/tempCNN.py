"""TemporalCNN classifier for downstream binary deforestation mapping."""

from pathlib import Path

import torch
import torch.nn as nn

from models.tempCNN import TempCNN


class TemporalCNN(nn.Module):
    """Fuse pretrained or randomly initialized S1/S2 TempCNN encoders."""

    def __init__(
        self,
        hidden_dim: int = 128,
        expander_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.modelname = (
            f"TemporalCNN_h={hidden_dim}_exp={expander_dim}_cls={num_classes}"
        )
        self.encoder_s1 = TempCNN(
            input_dim=2, kernel_size=7, hidden_dims=hidden_dim, dropout=0.5
        )
        self.encoder_s2 = TempCNN(
            input_dim=10, kernel_size=7, hidden_dims=hidden_dim, dropout=0.5
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, expander_dim),
            nn.ReLU(),
            nn.Linear(expander_dim, expander_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(expander_dim, num_classes)
        self._encoders_frozen = False

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        repr_s1 = self.encoder_s1(s1)
        repr_s2 = self.encoder_s2(s2)
        fused = torch.cat((repr_s1, repr_s2), dim=1)
        return self.classifier(self.head(fused))

    def load_pretrained_encoders(
        self, checkpoint_path: str, device: str | torch.device = "cpu"
    ) -> None:
        """Load only the S1/S2 encoders from a VICReg checkpoint."""
        checkpoint_path = str(Path(checkpoint_path))
        state = torch.load(checkpoint_path, map_location=device)
        full_state = state.get("model", state)

        s1_state = {
            key.removeprefix("encoder_s1."): value
            for key, value in full_state.items()
            if key.startswith("encoder_s1.")
        }
        s2_state = {
            key.removeprefix("encoder_s2."): value
            for key, value in full_state.items()
            if key.startswith("encoder_s2.")
        }

        if not s1_state or not s2_state:
            raise ValueError(
                "Checkpoint does not contain encoder_s1.* and encoder_s2.* weights"
            )

        self.encoder_s1.load_state_dict(s1_state, strict=True)
        self.encoder_s2.load_state_dict(s2_state, strict=True)

    def freeze_encoders(self) -> None:
        """Freeze feature encoders while leaving the downstream head trainable."""
        for parameter in self.encoder_s1.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder_s2.parameters():
            parameter.requires_grad = False
        self._encoders_frozen = True
        self.encoder_s1.eval()
        self.encoder_s2.eval()

    def train(self, mode: bool = True):
        """Keep frozen encoders in eval mode so BN/dropout do not drift."""
        super().train(mode)
        if self._encoders_frozen:
            self.encoder_s1.eval()
            self.encoder_s2.eval()
        return self
