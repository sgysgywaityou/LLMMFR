"""
Dual-attention mechanism with historical information enhancement
Sections 3.4.1 and 3.4.2
Equations 9-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism (Vaswani et al., 2017)
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, q_len, embed_dim)
            key: (batch_size, k_len, embed_dim)
            value: (batch_size, v_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: (batch_size, q_len, embed_dim)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.q_proj(query)  # (batch, q_len, embed_dim)
        K = self.k_proj(key)    # (batch, k_len, embed_dim)
        V = self.v_proj(value)  # (batch, v_len, embed_dim)

        # Reshape: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, q_len, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output


class TextDominatedAttention(nn.Module):
    """
    Attention mechanism dominated by news documents (Equations 9-10)
    Q = M_text, K = V = M_img / M_ded / M_red
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, M_text: torch.Tensor, M_img: torch.Tensor,
                M_ded: torch.Tensor, M_red: torch.Tensor) -> torch.Tensor:
        """
        Compute text-dominated attention and concatenate results (Equation 10)

        Args:
            M_text, M_img, M_ded, M_red: (batch_size, embed_dim)

        Returns:
            C1: (batch_size, embed_dim)
        """
        # M_t-i: text to image attention (Equation 9)
        M_t_i = self.mha(M_text.unsqueeze(1), M_img.unsqueeze(1), M_img.unsqueeze(1)).squeeze(1)

        # M_t-d: text to DED attention
        M_t_d = self.mha(M_text.unsqueeze(1), M_ded.unsqueeze(1), M_ded.unsqueeze(1)).squeeze(1)

        # M_t-r: text to RED attention
        M_t_r = self.mha(M_text.unsqueeze(1), M_red.unsqueeze(1), M_red.unsqueeze(1)).squeeze(1)

        # Concatenate and transform (Equation 10)
        concat = torch.cat([M_t_i, M_t_d, M_t_r], dim=-1)
        C1 = self.layer_norm(self.linear(concat))

        return C1


class ImageDominatedAttention(nn.Module):
    """
    Attention mechanism dominated by news images (Equation 11)
    Q = M_img / M_ded / M_red, K = V = M_text
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, M_text: torch.Tensor, M_img: torch.Tensor,
                M_ded: torch.Tensor, M_red: torch.Tensor) -> torch.Tensor:
        """
        Compute image-dominated attention and concatenate results (Equation 11)

        Args:
            M_text, M_img, M_ded, M_red: (batch_size, embed_dim)

        Returns:
            C2: (batch_size, embed_dim)
        """
        # M_i-t: image to text attention
        M_i_t = self.mha(M_img.unsqueeze(1), M_text.unsqueeze(1), M_text.unsqueeze(1)).squeeze(1)

        # M_d-t: DED to text attention
        M_d_t = self.mha(M_ded.unsqueeze(1), M_text.unsqueeze(1), M_text.unsqueeze(1)).squeeze(1)

        # M_r-t: RED to text attention
        M_r_t = self.mha(M_red.unsqueeze(1), M_text.unsqueeze(1), M_text.unsqueeze(1)).squeeze(1)

        # Concatenate and transform (Equation 11)
        concat = torch.cat([M_i_t, M_d_t, M_r_t], dim=-1)
        C2 = self.layer_norm(self.linear(concat))

        return C2


class HistoricalInformationEnhancement(nn.Module):
    """
    Historical information enhancement using residual connections (Equations 12-13)
    """

    def __init__(self, embed_dim: int, text_dim: int, image_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.image_proj = nn.Linear(image_dim, embed_dim)

    def forward(self, C1: torch.Tensor, C2: torch.Tensor,
                DV_text: torch.Tensor, PV_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate historical information (Equations 12-13)

        Args:
            C1, C2: (batch_size, embed_dim)
            DV_text: (batch_size, seq_len, text_dim)
            PV_img: (batch_size, num_patches, image_dim)

        Returns:
            C1_prime, C2_prime: (batch_size, embed_dim)
        """
        # Pool original features to get sequence-level representation
        DV_pooled = DV_text.mean(dim=1)  # (batch, text_dim)
        PV_pooled = PV_img.mean(dim=1)   # (batch, image_dim)

        # Project to embed_dim
        DV_proj = self.text_proj(DV_pooled)
        PV_proj = self.image_proj(PV_pooled)

        # Historical information integration (Equations 12-13)
        C1_prime = C1 + self.layer_norm(DV_proj)
        C2_prime = C2 + self.layer_norm(PV_proj)

        return C1_prime, C2_prime


class MainClassifier(nn.Module):
    """
    Main classifier combining dual-attention and historical information (Sections 3.4.1-3.4.2)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_dominated_attn = TextDominatedAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        self.image_dominated_attn = ImageDominatedAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        self.historical_enhancement = HistoricalInformationEnhancement(
            embed_dim=config.hidden_dim,
            text_dim=config.text_dim,
            image_dim=config.image_dim
        )

    def forward(self, M_text: torch.Tensor, M_img: torch.Tensor,
                M_ded: torch.Tensor, M_red: torch.Tensor,
                DV_text: torch.Tensor, PV_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            C1: text-dominated attention output
            C2: image-dominated attention output
            C1_prime: C1 with historical information (Equation 12)
            C2_prime: C2 with historical information (Equation 13)
        """
        # Dual-attention (Equations 10-11)
        C1 = self.text_dominated_attn(M_text, M_img, M_ded, M_red)
        C2 = self.image_dominated_attn(M_text, M_img, M_ded, M_red)

        # Historical information enhancement (Equations 12-13)
        C1_prime, C2_prime = self.historical_enhancement(C1, C2, DV_text, PV_img)

        return C1, C2, C1_prime, C2_prime