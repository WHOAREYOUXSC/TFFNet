from torch import nn

from TFFNet.models.blocks.freq_block import Frequency_Block
from TFFNet.models.blocks.time_block import Time_Block
from TFFNet.models.layers.crc import ChannelReweightConcat


class TimeFrequencyAttentionFusion(nn.Module):
    """
    Time-Frequency Attention Fusion (TFAF)

    Combines:
    - Frequency branch (FFCM)
    - Time branch (PPA)

    Followed by channel-wise reweighting and concatenation (CRC)
    """

    def __init__(self, in_channels, out_channels):
        super(TimeFrequencyAttentionFusion, self).__init__()

        self.freq_branch = Frequency_Block(in_channels=in_channels, out_channels=out_channels)
        self.time_branch = Time_Block(in_features=in_channels, filters=out_channels)

    def forward(self, x):

        # Frequency features
        x_freq = self.freq_branch(x)

        # Time-domain features
        x_time = self.time_branch(x)

        # Channel-aware fusion
        out = [x_freq, x_time]

        return out