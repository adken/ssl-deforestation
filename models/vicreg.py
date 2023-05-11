import torch
from models.tempCNN import TempCNN as TempCNNEncoder
import torch.nn as nn


class VICRegNet(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 expander_dim=256):
        super().__init__()
        self.encoder_s1 = TempCNNEncoder(input_dim=2, kernel_size=5, hidden_dims=hidden_dim, dropout=0.5)
        self.encoder_s2 = TempCNNEncoder(input_dim=10, kernel_size=5, hidden_dims=hidden_dim, dropout=0.5)

        self.expander = nn.Sequential(
            nn.Linear(hidden_dim, expander_dim),
            nn.BatchNorm1d(expander_dim),
            nn.ReLU(),
            nn.Linear(expander_dim, expander_dim),
            nn.BatchNorm1d(expander_dim),
            nn.ReLU(),
            nn.Linear(expander_dim, expander_dim))

    def forward(self, s1, s2):
        _repr_s1 = self.encoder_s1(s1)
        _repr_s2 = self.encoder_s2(s2)

        _embeds_1 = self.expander(_repr_s1.squeeze())
        _embeds_2 = self.expander(_repr_s2.squeeze())

        return _embeds_1, _embeds_2



if __name__ == '__main__':
    model = VICRegNet()
