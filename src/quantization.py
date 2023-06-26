import torch
import torch.nn as nn


class NormNoiseQuantization(nn.Module):
    def __init__(self, quants: int = 2):
        super().__init__()
        self.quants = quants

    def forward(self, x):
        x = x + ((torch.rand_like(x, requires_grad=True) - 0.5) / (2 ** self.quants))
        return x
