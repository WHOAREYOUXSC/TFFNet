import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalAttention(nn.Module):
    """Patch(Local-global) attention with patch-based modeling."""

    def __init__(self, output_dim, patch_size):
        super().__init__()

        self.output_dim = output_dim
        self.patch_size = patch_size

        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)

        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        self.prompt = nn.Parameter(torch.randn(output_dim))
        self.top_down_transform = nn.Parameter(torch.eye(output_dim))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        patches = x.unfold(1, P, P).unfold(2, P, P)
        patches = patches.reshape(B, -1, P * P, C)
        patches = patches.mean(dim=-1)

        patches = self.mlp1(patches)
        patches = self.norm(patches)
        patches = self.mlp2(patches)

        attention = F.softmax(patches, dim=-1)
        local_out = patches * attention

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        mask = cos_sim.clamp(0, 1)

        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)
        local_out = local_out.permute(0, 3, 1, 2)

        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)

        return self.conv(local_out)