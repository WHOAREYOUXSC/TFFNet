import torch
import torch.nn as nn


class SpatialAttentionModule(nn.Module):
    """Spatial attention using avg + max pooling."""

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv2d(out))

        return attention * x