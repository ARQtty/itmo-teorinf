import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder


class AEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)
