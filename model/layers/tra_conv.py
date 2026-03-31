from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, act=nn.LeakyReLU(0.1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2),
            nn.BatchNorm2d(out_ch),
            act
        )

    def forward(self, x):
        return self.block(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)