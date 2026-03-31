import torch
import torch.nn as nn



class ChannelReweightConcat(nn.Module):
    """
    Channel Reweighting and Concatenation (CRC)

    Applies learnable channel-wise weights to two feature maps
    before concatenation.
    """

    def __init__(self, dim=1, in_channels1=1, in_channels2=1):
        super(ChannelReweightConcat, self).__init__()

        self.dim = dim
        self.total_channels = in_channels1 + in_channels2

        self.weights = nn.Parameter(
            torch.ones(self.total_channels, dtype=torch.float32),
            requires_grad=True
        )

        self.eps = 1e-4

    def forward(self, x1, x2):

        N1, C1, H1, W1 = x1.size()
        N2, C2, H2, W2 = x2.size()

        x = torch.cat([x1, x2], dim=1)  # [B, C1+C2, H, W]

        # Normalize weights
        w = self.weights[:C1 + C2]
        w = w / (torch.sum(w) + self.eps)

        # Apply channel-wise weighting
        x1 = (w[:C1] * x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = (w[C1:] * x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Concatenate
        out = torch.cat([x1, x2], dim=self.dim)

        return out