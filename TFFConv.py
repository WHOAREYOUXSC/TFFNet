import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class DynamicFrequencyDirectionalComponentsFeatureProcessor(nn.Module):
    """
    Dynamic Kernel Feature Processor with Wavelet Decomposition and Frequency-Attention

    This module processes input features through:
    1. Wavelet decomposition to extract multi-directional frequency components
    2. Learnable channel-wise weights for horizontal/vertical/diagonal details
    3. Dynamic kernel size convolution adapted to input characteristics
    4. Squeeze-and-Excitation channel attention mechanism
    5. Wavelet reconstruction to restore spatial dimensions

    Args:
        input_channel_count (int): Number of input channels
        output_channel_count (int): Number of output channels
    """

    def __init__(self, input_channel_count, output_channel_count):
        super(DynamicFrequencyDirectionalComponentsFeatureProcessor, self).__init__()

        # Wavelet transform components
        self.discrete_wavelet_transform = DWTForward(J=1, mode='zero', wave='haar')
        self.inverse_dwt = DWTInverse(mode='zero', wave='haar')

        # Feature fusion block (Conv + BN + Activation)
        self.convolution_batchnorm_activation = nn.Sequential(
            nn.Conv2d(input_channel_count * 4, output_channel_count * 4,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channel_count * 4),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Dynamic convolution components
        self.dynamic_kernel_size = nn.Parameter(torch.randn(1))  # Learnable kernel size
        self.conv_for_dynamic_kernel = nn.Conv2d(
            input_channel_count * 4,
            output_channel_count * 4,
            kernel_size=3, stride=1, padding=1)

        # Channel attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channel_count * 4, output_channel_count, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(output_channel_count, output_channel_count * 4, kernel_size=1),
            nn.Sigmoid()
        )

        # Learnable channel-wise weights for directional components
        self.horizontal_weight = nn.Parameter(torch.randn(1, input_channel_count, 1, 1))
        self.vertical_weight = nn.Parameter(torch.randn(1, input_channel_count, 1, 1))
        self.diagonal_weight = nn.Parameter(torch.randn(1, input_channel_count, 1, 1))

    def dynamic_convolution(self, x):
        """
        Performs convolution with dynamically learned kernel size

        Args:
            x (torch.Tensor): Input feature map

        Returns:
            torch.Tensor: Feature map after dynamic convolution
        """
        device = x.device
        # Generate odd kernel size based on learned parameter
        kernel_size = int(torch.clamp(self.dynamic_kernel_size, min=1).item())
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        padding = kernel_size // 2

        # Create depthwise convolution with dynamic kernel size
        dynamic_conv = nn.Conv2d(
            x.size(1), x.size(1),
            kernel_size=kernel_size,
            padding=padding,
            groups=x.size(1),
            bias=False).to(device)
        dynamic_conv.weight.data.fill_(1.0)  # Initialize with uniform weights

        return dynamic_conv(x)

    def forward(self, input_tensor):
        """
        Forward pass through the feature processor

        Args:
            input_tensor (torch.Tensor): Input feature map of shape [B, C, H, W]

        Returns:
            torch.Tensor: Processed feature map of shape [B, C', H, W]
        """
        # Wavelet decomposition into frequency components
        low_freq, high_freqs = self.discrete_wavelet_transform(input_tensor)

        # Extract directional components (horizontal, vertical, diagonal)
        horizontal = high_freqs[0][:, :, 0, :, :]  # Horizontal detail coefficients
        vertical = high_freqs[0][:, :, 1, :, :]  # Vertical detail coefficients
        diagonal = high_freqs[0][:, :, 2, :, :]  # Diagonal detail coefficients

        # Apply learned channel-wise weights to directional components
        horizontal = horizontal * self.horizontal_weight
        vertical = vertical * self.vertical_weight
        diagonal = diagonal * self.diagonal_weight

        # Feature fusion: concatenate all frequency components
        merged_features = torch.cat([
            low_freq,  # Approximation coefficients
            horizontal,  # Weighted horizontal details
            vertical,  # Weighted vertical details
            diagonal  # Weighted diagonal details
        ], dim=1)

        # Dynamic convolution processing
        merged_features = self.dynamic_convolution(merged_features)

        # 4. Feature refinement with conv+bn+activation
        fused_features = self.convolution_batchnorm_activation(merged_features)

        # Channel attention (recalibration)
        attention = self.channel_attention(fused_features)
        fused_features = attention * fused_features

        # Prepare components for inverse wavelet transform
        C = fused_features.shape[1] // 4
        new_low = fused_features[:, :C, :, :]  # New approximation coefficients
        new_high = torch.stack([
            fused_features[:, C:C * 2, :, :],  # New horizontal details
            fused_features[:, C * 2:C * 3, :, :],  # New vertical details
            fused_features[:, C * 3:C * 4, :, :]  # New diagonal details
        ], dim=2)

        # Wavelet reconstruction to original spatial dimensions
        output = self.inverse_dwt((new_low, [new_high]))

        return output

class FrequencyFeatureProcessor(nn.Module):
    """
    Pure Wavelet-based Feature Processing Module:
    - Multi-scale Discrete Wavelet Transform (DWT)
    - Directional feature processing (horizontal/vertical/diagonal)
    - Dynamic kernel convolution
    - Channel attention refinement

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        groups (int): Groups for grouped convolution (default=1)
    """

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        # Initial feature projection
        self.init_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Multi-scale processing branches
        self.branch3x3 = self._build_conv_branch(kernel_size=3)
        self.branch5x5 = self._build_conv_branch(kernel_size=5)

        # Wavelet processor
        self.wavelet_processor = DynamicFrequencyDirectionalComponentsFeatureProcessor(
            input_channel_count=out_channels * 2,
            output_channel_count=out_channels
        )

        self.ca_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, groups=self.out_channels, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=1, groups=out_channels,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.1)
        )

    def _build_conv_branch(self, kernel_size):
        """Build multi-scale convolution branch"""
        return nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size,
                      padding=kernel_size // 2, groups=self.out_channels,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.out_channels, self.out_channels, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # Initial projection
        x_proj = self.init_proj(x)

        # Multi-scale processing
        x1 = self.branch3x3(x_proj)
        x2 = self.branch5x5(x_proj)

        # Wavelet processing
        x_cat = torch.cat([x1, x2], dim=1)
        wavelet_out = self.wavelet_processor(x_cat)

        # Feature fusion with residual
        fused = wavelet_out + x_proj

        # Channel attention
        attn = self.ca_conv(fused)
        return self.refine(fused * attn)

class SpatialChannelAttention(nn.Module):
    """Combined spatial and channel attention mechanism"""

    def __init__(self, in_channel):
        super().__init__()
        # Spatial attention (using max and avg pooling)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # Processes concatenated features
            nn.Sigmoid()
        )

        # Channel attention (ECA module)
        gamma, b = 2, 1  # Hyperparameters from ECA paper
        k_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1  # Ensure odd kernel size

        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial attention computation
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))

        # Channel attention computation
        channel_att = self.channel(x.unsqueeze(1)).squeeze(1)

        return x * spatial_att * channel_att  # Apply both attentions

class LocalGlobalAttention(nn.Module):
    """Attention mechanism combining local patch processing with global context"""

    def __init__(self, dim, patch_size):
        super().__init__()
        self.patch_size = patch_size

        # MLP for patch feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(patch_size * patch_size, dim // 2),  # Dimension reduction
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, dim)  # Dimension restoration
        )

        # Projection and prompt learning
        self.proj = nn.Conv2d(dim, dim, 1)  # Final 1x1 conv
        self.prompt = nn.Parameter(torch.randn(dim))  # Learnable prompt vector
        self.transform = nn.Parameter(torch.eye(dim))  # Learnable transformation

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Extract and process local patches
        patches = x.unfold(2, P, P).unfold(3, P, P)  # [B,C,H/P,W/P,P,P]
        patches = patches.reshape(B, C, -1, P * P).permute(0, 2, 3, 1)  # [B,N,P*P,C]
        patches = self.mlp(patches.mean(dim=-2))  # [B,N,C]

        # Compute similarity with learned prompt
        sim = F.cosine_similarity(patches, self.prompt[None, None, :], dim=-1).clamp(0, 1)
        patches = patches * sim.unsqueeze(-1) @ self.transform  # Apply transformation

        # Reconstruct feature map
        out = patches.permute(0, 2, 1).reshape(B, C, H // P, W // P)
        out = F.interpolate(out, size=(H, W), mode='bilinear')  # Upsample to original size
        return self.proj(out)

class TimeFeatureProcessor(nn.Module):
    """Multi-scale temporal feature processor with attention"""

    def __init__(self, in_features, filters):
        super().__init__()
        # Skip connection path
        self.skip = self._build_conv(in_features, filters, 1, 0, activation=False)

        # Multi-scale convolutional path (3 layers)
        self.conv_path = nn.Sequential(
            self._build_conv(in_features, filters, 3, 1),  # conv1
            self._build_conv(filters, filters, 3, 1),  # conv2
            self._build_conv(filters, filters, 3, 1)  # conv3
        )

        # Attention modules
        self.attention = SpatialChannelAttention(filters)  # Dual attention
        self.lga2 = LocalGlobalAttention(filters, 2)  # Local-global attention (small patches)
        self.lga4 = LocalGlobalAttention(filters, 4)  # Local-global attention (large patches)

        # Output processing
        self.norm = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout2d(0.1)  # Regularization
        self.act = nn.GELU()  # Activation

    def _build_conv(self, in_c, out_c, ks, pad, activation=True):
        """Helper to build conv-bn-relu blocks"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU() if activation else nn.Identity()
        )

    def forward(self, x):
        # Residual connection
        identity = self.skip(x)

        # Multi-scale feature extraction
        x1 = self.conv_path[0](x)  # First conv
        x2 = self.conv_path[1](x1)  # Second conv
        x3 = self.conv_path[2](x2)  # Third conv

        # Feature fusion (sum all paths)
        x = x1 + x2 + x3 + identity

        # Apply attention mechanisms
        x = self.attention(x)  # Spatial-channel attention
        x = x + self.lga2(x) + self.lga4(x)  # Add local-global attention features

        # Final processing
        return self.act(self.norm(self.dropout(x)))

# Power Average Pooling Layer definition
class PowerAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, p=2):
        super(PowerAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p  # The power to which to raise each element before averaging

    def forward(self, x):
        # Apply power (raise to the power of p) to the input
        x = torch.pow(x, self.p)  # Raise each element to the power of p
        # Apply average pooling
        x = nn.functional.avg_pool2d(x, self.kernel_size, self.stride)
        # Apply inverse power to return to the original scale
        x = torch.pow(x, 1 / self.p)  # Inverse power to recover from the raised value
        return x

class PowerAveragedChannelReweightingFusion(nn.Module):
    """Feature fusion with channel weighting -> 1x1 conv -> power avg pooling: PACF"""
    def __init__(self, dimension=1, channels1=1, channels2=1,
                 pool_stride=2, pool_kernel=2, power=2):
        super().__init__()

        # Configuration
        self.dim = dimension  # Default: channel-wise concatenation
        total_channels = channels1 + channels2

        # Feature processing
        self.conv_reduce = nn.Conv2d(total_channels, channels1, kernel_size=1)  # Bottleneck
        self.pool = PowerAvgPool2d(pool_kernel, pool_stride, power)

        # Adaptive weighting
        self.weights = nn.Parameter(torch.ones(total_channels))
        self.eps = 1e-6

    def forward(self, features):
        # Input validation
        assert len(features) == 2, "Requires exactly two feature maps"
        x1, x2 = features

        # Channel-wise adaptive fusion
        weights = F.softmax(self.weights, dim=0)
        c1 = x1.size(1)
        fused = torch.cat([
            x1 * weights[:c1].view(1, -1, 1, 1),
            x2 * weights[c1:].view(1, -1, 1, 1)
        ], dim=self.dim)

        # Dimensionality reduction before pooling
        fused = self.conv_reduce(fused)
        return self.pool(fused)

class TFFConv(nn.Module):
    """Two-path feature fusion module with time and frequency processing"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Time feature processor
        self.time_path = TimeFeatureProcessor(in_features=in_channels,
                                              filters=out_channels)
        # Frequency feature processor
        self.freq_path = FrequencyFeatureProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=1
        )
        # Feature fusion module
        self.fusion = PowerAveragedChannelReweightingFusion(
            dimension=1,
            channels1=out_channels,
            channels2=out_channels,
            pool_stride=1,  # No spatial reduction
            pool_kernel=1
        )

    def forward(self, x):
        # Process through both paths
        time_feat = self.time_path(x)
        freq_feat = self.freq_path(x)

        # Fuse features
        return self.fusion([freq_feat, time_feat])


def test_TFFConv():
    # Setup parameters
    in_channels = 64
    out_channels = 128
    batch_size = 4
    height = width = 32

    # Initialize model
    model = TFFConv(in_channels, out_channels)

    # Create dummy input
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    try:
        # Forward pass test
        output = model(dummy_input)
        assert output.shape == (batch_size, out_channels, height, width), \
            f"Shape mismatch. Expected: {(batch_size, out_channels, height, width)}, Got: {output.shape}"

        # Backward pass test
        target = torch.randn_like(output)
        loss = torch.mean((output - target)**2)
        loss.backward()

        print("Test passed! Module working correctly")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"Test failed! Error: {str(e)}")
        return False

test_TFFConv()