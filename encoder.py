import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = 2
        self.model = nn.Sequential(
            self.conv_block(3, base ** 4, 3),
            self.conv_block(base ** 4, base ** 5, 3),
            self.conv_block(base ** 5, base ** 6, 5),
            self.conv_block(base ** 6, base ** 7, 5),
            nn.Conv2d(base ** 7, base ** 8, kernel_size=4),
            nn.LeakyReLU(0.1)
        )

    def conv_block(self, dim_in: int, dim_out: int, kernel_size: int, **kwargs) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, **kwargs),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(dim_out),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(3)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
