from torch import nn

from TFFNet.models.layers.crc import ChannelReweightConcat
from TFFNet.models.layers.pap import PowerAvgPool


class PowerAveragedChannelFusion(nn.Module):
    """
    Power-Averaged Channel Fusion (PACF)

    Combines:
    - ChannelReweightConcat (CRC)
    - PowerAvgPool (PAP)

    Pipeline:
        (x1, x2) → CRC → PAP → Output
    """

    def __init__(
        self,
        in_channels1,
        in_channels2,
        kernel_size=2,
        stride=2,
        p=2.0
    ):
        super(PowerAveragedChannelFusion, self).__init__()

        self.crc = ChannelReweightConcat(
            dim=1,
            in_channels1=in_channels1,
            in_channels2=in_channels2
        )

        self.pap = PowerAvgPool(
            kernel_size=kernel_size,
            stride=stride,
            p=p
        )

        self.conv1 = nn.Conv2d(in_channels1 * 2, in_channels1, kernel_size=1)

    def forward(self, x1, x2):

        x = self.crc(x1, x2)
        x = self.conv1(x)
        x = self.pap(x)

        return x