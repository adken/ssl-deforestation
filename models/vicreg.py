import torch
#from torchvision.models import resnet50
from tempCNN import TempCNN
import torch.nn as nn


class VICRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.encoder = resnet50(pretrained=False)
        self.encoder = TempCNN()
        #self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]),
        #                           nn.Flatten())

        self.expander = nn.Sequential(
            #nn.Linear(2048, 8192),
            nn.Linear(128, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192))

    def forward(self, x):
        _repr = self.encoder(x)
        _embeds = self.expander(_repr.squeeze())
        #_embeds = self.expander(_repr)
        return _embeds


if __name__ == '__main__':
    model = VICRegNet()
