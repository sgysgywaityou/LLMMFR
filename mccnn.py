"""
Multi-Channel Convolutional Neural Network (Section 3.3)
Equations 5-8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SingleChannelConv1D(nn.Module):
    """
    Single-channel 1D convolution for MCCNN
    """

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size // 2)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            out: (batch_size, output_dim, seq_len)
        """
        # Transpose to (batch, input_dim, seq_len) for Conv1D
        x = x.transpose(1, 2)
        out = self.conv(x) + self.bias.view(1, -1, 1)
        return out


class MultiChannelExtractor(nn.Module):
    """
    Multi-Channel Convolutional Neural Network (MCCNN)
    Extracts multi-view representations for each modality
    """

    def __init__(self, input_dim: int, num_channels: int, output_dim: int, kernel_size: int = 3):
        super().__init__()
        self.num_channels = num_channels
        self.output_dim = output_dim

        # Create multiple single-channel convolutions
        self.convs = nn.ModuleList([
            SingleChannelConv1D(input_dim, output_dim, kernel_size)
            for _ in range(num_channels)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) or (batch_size, input_dim) for pooled features

        Returns:
            output: (batch_size, output_dim)
        """
        # Handle 2D input (already pooled)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)

        # Original shape: (batch, seq_len, input_dim)
        batch_size = x.size(0)

        channel_outputs = []

        for conv in self.convs:
            # conv output: (batch, output_dim, seq_len)
            conv_out = conv(x)

            # Apply softmax (Equation 5)
            # Softmax over the channel dimension
            conv_out = self.softmax(conv_out)

            # Max pooling over sequence dimension
            # (batch, output_dim, seq_len) -> (batch, output_dim)
            pooled = torch.max(conv_out, dim=2)[0]

            channel_outputs.append(pooled)

        # Sum over channels (Equation 5)
        # (num_channels, batch, output_dim) -> (batch, output_dim)
        output = torch.stack(channel_outputs, dim=0).sum(dim=0)

        return output


class MultiChannelExtractorWithAttention(nn.Module):
    """
    Enhanced MCCNN with attention-based channel weighting
    """

    def __init__(self, input_dim: int, num_channels: int, output_dim: int, kernel_size: int = 3):
        super().__init__()
        self.base_extractor = MultiChannelExtractor(input_dim, num_channels, output_dim, kernel_size)

        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Linear(output_dim * num_channels, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, num_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = x.size(0)
        channel_outputs = []

        # Get individual channel outputs
        for i, conv in enumerate(self.base_extractor.convs):
            conv_out = conv(x)
            conv_out = self.base_extractor.softmax(conv_out)
            pooled = torch.max(conv_out, dim=2)[0]
            channel_outputs.append(pooled)

        # Stack channel outputs: (batch, output_dim, num_channels)
        stacked = torch.stack(channel_outputs, dim=2)

        # Compute channel attention weights
        # Flatten for attention: (batch, output_dim * num_channels)
        flat = stacked.view(batch_size, -1)
        attn_weights = self.channel_attention(flat)  # (batch, num_channels)

        # Apply attention weights
        # (batch, output_dim, num_channels) * (batch, 1, num_channels) -> (batch, output_dim)
        weighted = (stacked * attn_weights.unsqueeze(1)).sum(dim=2)

        return weighted


class MultiChannelFeatureExtractor(nn.Module):
    """
    Extract multi-channel features for all modalities
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # MCCNN for text modality (Equation 5)
        self.mccnn_text = MultiChannelExtractor(
            input_dim=config.text_dim,
            num_channels=config.num_text_channels,
            output_dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size
        )

        # MCCNN for image modality (Equation 6)
        self.mccnn_img = MultiChannelExtractor(
            input_dim=config.image_dim,
            num_channels=config.num_image_channels,
            output_dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size
        )

        # MCCNN for DED modality (Equation 7)
        self.mccnn_ded = MultiChannelExtractor(
            input_dim=config.text_dim,
            num_channels=config.num_ded_channels,
            output_dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size
        )

        # MCCNN for RED modality (Equation 8)
        self.mccnn_red = MultiChannelExtractor(
            input_dim=config.text_dim,
            num_channels=config.num_red_channels,
            output_dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size
        )

    def forward(self, DV_text: torch.Tensor, PV_img: torch.Tensor,
                PV_ded: torch.Tensor, PV_red: torch.Tensor) -> tuple:
        """
        Args:
            DV_text: (batch_size, seq_len_text, text_dim)
            PV_img: (batch_size, num_patches, image_dim)
            PV_ded: (batch_size, seq_len_ded, text_dim)
            PV_red: (batch_size, seq_len_red, text_dim)

        Returns:
            M_text, M_img, M_ded, M_red: each (batch_size, hidden_dim)
        """
        M_text = self.mccnn_text(DV_text)  # Equation 5
        M_img = self.mccnn_img(PV_img)  # Equation 6
        M_ded = self.mccnn_ded(PV_ded)  # Equation 7
        M_red = self.mccnn_red(PV_red)  # Equation 8

        return M_text, M_img, M_ded, M_red