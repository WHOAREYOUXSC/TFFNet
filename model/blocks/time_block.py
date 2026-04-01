import torch.nn as nn

from TFFNet.models.attention.spatial import SpatialAttentionModule
from TFFNet.models.attention.eca import ECA
from TFFNet.models.attention.local_global import LocalGlobalAttention


class Time_Block(nn.Module):
    """
    Time branch module.

    Structure:
        Multi-branch conv + skip + local-global attention
        → Channel attention (ECA)
        → Spatial attention
        → Dropout + BN + Activation
    """

    def __init__(self, in_features, filters):
        super().__init__()

        # ===== Skip branch (1×1 conv) =====
        self.skip = nn.Sequential(
            nn.Conv2d(in_features, filters, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(filters)
        )

        # ===== Convolution branches =====
        self.c1 = nn.Sequential(
            nn.Conv2d(in_features, filters, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=False)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=False)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=False)
        )

        # ===== Attention modules =====
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.channel_attn = ECA(filters)
        self.spatial_attn = SpatialAttentionModule()

        # ===== Output processing =====
        self.dropout = nn.Dropout2d(0.1)
        self.bn = nn.BatchNorm2d(filters)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):

        # ===== Skip =====
        x_skip = self.skip(x)

        # ===== Local-global attention =====
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)

        # ===== Conv branches =====
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        # ===== Multi-branch fusion =====
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4

        # ===== Attention refinement =====
        x = self.channel_attn(x)
        x = self.spatial_attn(x)

        # ===== Output =====
        x = self.dropout(x)
        x = self.bn(x)
        x = self.act(x)

        return x