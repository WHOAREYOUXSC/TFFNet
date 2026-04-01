import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class WaveletDynamicConv(nn.Module):
    """
    Dynamic Frequency-Domain Directional Components Module.

    Components:
    - Discrete Wavelet Transform (DWT)
    - Direction-aware feature reweighting
    - Dynamic depthwise convolution (kernel selection)
    - Channel attention (SE)
    - Inverse DWT reconstruction
    """

    def __init__(self, in_channels, out_channels, min_kernel_size=3, max_kernel_size=9):
        super(WaveletDynamicConv, self).__init__()

        # ===== Wavelet Transform =====
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')

        # ===== Post-conv Fusion =====
        self.post_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels * 4, kernel_size=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.LeakyReLU(0.1)
        )

        # ===== Dynamic Kernel Selection =====
        self.min_k = min_kernel_size
        self.max_k = max_kernel_size

        kernel_list = list(range(min_kernel_size, max_kernel_size + 1, 2))
        self.num_kernels = len(kernel_list)

        self.kernel_logits = nn.Parameter(torch.randn(self.num_kernels))

        self.dynamic_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels * 4,
                in_channels * 4,
                kernel_size=k,
                padding=k // 2,
                groups=in_channels * 4,
                bias=False
            ) for k in kernel_list
        ])

        # ===== Channel Attention =====
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 4, out_channels * 4 // 16, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels * 4 // 16, out_channels * 4, kernel_size=1),
            nn.Sigmoid()
        )

        # ===== Directional Weights =====
        self.weight_h = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.weight_v = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.weight_d = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def _select_kernel(self):
        """
        Select kernel size via softmax / Gumbel-softmax
        """
        probs = F.softmax(self.kernel_logits, dim=0)

        if self.training:
            gumbel = F.gumbel_softmax(self.kernel_logits, tau=1.0, hard=True)
            idx = torch.argmax(gumbel).item()
        else:
            idx = torch.argmax(probs).item()

        kernel_size = self.min_k + idx * 2
        return kernel_size, probs

    def _dynamic_conv(self, x):
        """
        Apply dynamic depthwise convolution
        """
        kernel_size, probs = self._select_kernel()

        idx = (kernel_size - self.min_k) // 2
        idx = min(idx, len(self.dynamic_convs) - 1)

        out = self.dynamic_convs[idx](x)

        if self.training:
            fused = 0
            for i, conv in enumerate(self.dynamic_convs):
                fused += conv(x) * probs[i]
            out = fused

        return out, kernel_size

    def forward(self, x):

        # ===== DWT Decomposition =====
        low, high = self.dwt(x)

        h = high[0][:, :, 0, :, :]
        v = high[0][:, :, 1, :, :]
        d = high[0][:, :, 2, :, :]

        # ===== Directional Reweighting =====
        h = h * self.weight_h
        v = v * self.weight_v
        d = d * self.weight_d

        # ===== Merge Subbands =====
        x = torch.cat([low, h, v, d], dim=1)

        # ===== Dynamic Convolution =====
        x, k = self._dynamic_conv(x)
        self.current_kernel_size = k

        # ===== Feature Fusion =====
        x = self.post_conv(x)

        # ===== Channel Attention =====
        att = self.channel_attention(x)
        x = x * att

        # ===== Split for IDWT =====
        C = x.shape[1] // 4
        low = x[:, :C, :, :]
        high = torch.stack([
            x[:, C:2*C, :, :],
            x[:, 2*C:3*C, :, :],
            x[:, 3*C:4*C, :, :]
        ], dim=2)

        # ===== Reconstruction =====
        out = self.idwt((low, [high]))

        return out

    def get_kernel_info(self):
        """
        Return current kernel selection statistics
        """
        if hasattr(self, 'current_kernel_size'):
            probs = F.softmax(self.kernel_logits, dim=0)
            return {
                'current_kernel_size': self.current_kernel_size,
                'kernel_probabilities': probs.detach().cpu().numpy()
            }
        return None