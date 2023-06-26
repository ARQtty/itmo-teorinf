import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            self.conv_block(256, 128, 2, 2 ,),
            self.conv_block(128, 64, 3, 2 ,),

            self.upscale_block(64, 64, 4),
            self.upscale_block(64, 32, 4),
            self.upscale_block(32, 16, 4),
            self.upscale_block(16, 3, 2),
        )

    def conv_block(self, dim_in: int, dim_out: int, kernel_size: int, stride: int, padding: int = 0) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.BatchNorm2d(dim_out),
        )

    def upscale_block(self, dim_in: int, dim_out: int, upscale: int = 2) -> nn.Sequential:
        return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=upscale),
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dim_out),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
