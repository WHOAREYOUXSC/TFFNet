import torch
from torch import nn

from TFFNet.models.tfaf_module import TimeFrequencyAttentionFusion
from TFFNet.models.layers.tra_conv import DoubleConv
from TFFNet.models.layers.crc import ChannelReweightConcat
from TFFNet.models.layers.pap import PowerAvgPool


class TFFConv(nn.Module):
    """TFAF + PACF block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Time-Frequency Attention Fusion
        self.TFAF = TimeFrequencyAttentionFusion(in_channels, out_channels)

        self.CRC = ChannelReweightConcat(dim=1, in_channels1=out_channels, in_channels2=out_channels)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x_freq, x_time = self.TFAF(x)
        out = self.CRC(x_freq, x_time)
        out = self.conv1(out)
        return out


class TFFNet(nn.Module):
    """TFFNet: Encoder-Decoder with TFFConv."""

    def __init__(self, in_channels=1, out_channels=1):
        super(TFFNet, self).__init__()

        self.channels = [64, 128, 256, 512]
        self.pool = PowerAvgPool(kernel_size=2, stride=2, p=2)

        # Encoder
        self.stage1 = TFFConv(in_channels, self.channels[0])
        self.stage2 = TFFConv(self.channels[0], self.channels[1])
        self.stage3 = TFFConv(self.channels[1], self.channels[2])

        self.enc4 = DoubleConv(self.channels[2], self.channels[3])

        # Decoder
        self.up3 = nn.ConvTranspose2d(self.channels[3], self.channels[2], 2, 2)
        self.dec3 = DoubleConv(self.channels[2] * 2, self.channels[2])

        self.up2 = nn.ConvTranspose2d(self.channels[2], self.channels[1], 2, 2)
        self.dec2 = DoubleConv(self.channels[1] * 2, self.channels[1])

        self.up1 = nn.ConvTranspose2d(self.channels[1], self.channels[0], 2, 2)
        self.dec1 = DoubleConv(self.channels[0] * 2, self.channels[0])

        # Output
        self.out_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.stage1(x)
        e2 = self.stage2(self.pool(e1))
        e3 = self.stage3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.out_conv(d1)
        out = self.activation(out)

        return out