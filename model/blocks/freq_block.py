import torch
import torch.nn as nn
from TFFNet.models.layers.dfdc import WaveletDynamicConv


class Frequency_Block(nn.Module):
    """
    Frequency-aware Feature Fusion Block

    Combines:
    - Local spatial feature extraction
    - Global frequency-aware representation (via DWT + dynamic convolution)
    """

    def __init__(self, in_channels, out_channels):
        super(Frequency_Block, self).__init__()

        # Channel projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        # Local branch (multi-scale depthwise conv)
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, groups=out_channels, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        self.local_5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5,
                      padding=2, groups=out_channels, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        # Pre-fusion branches
        self.branch_spatial = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        self.branch_context = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        # Frequency-domain processor (DFDC)
        self.freq_processor = WaveletDynamicConv(
            in_channels=out_channels * 2,
            out_channels=out_channels
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, groups=out_channels, padding_mode='reflect'),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):

        # Projection
        x = self.proj(x)

        # Local feature extraction
        x_l1 = self.local_3x3(x)
        x_l2 = self.local_5x5(x)

        # Split into two branches
        x_spatial = self.branch_spatial(x_l1)
        x_context = self.branch_context(x_l2)

        # Concatenate for frequency processing
        x_cat = torch.cat([x_spatial, x_context], dim=1)

        # Global frequency enhancement
        x_freq = self.freq_processor(x_cat)

        # Residual fusion
        out = x + x_freq

        # Refinement
        out = self.refine(out)

        return out