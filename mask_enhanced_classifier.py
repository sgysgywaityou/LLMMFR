"""
Mask Enhanced Classifier with progressive learnable masking and hybrid encoder-decoder
Section 3.4.3
Equations 14-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax for differentiable discrete sampling (Equation 15)
    """

    def __init__(self, tau: float = 1.0, hard: bool = False):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
        Returns:
            y: (batch_size, num_classes) soft/hard samples
        """
        # Add Gumbel noise
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / self.tau
        y_soft = F.softmax(gumbels, dim=-1)

        if self.hard:
            # Straight-through estimator
            index = y_soft.argmax(dim=-1)
            y_hard = torch.zeros_like(logits).scatter_(-1, index.unsqueeze(-1), 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret


class ProgressiveLearnableMask(nn.Module):
    """
    Progressive Learnable Mask mechanism (Equations 14-16)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, tau: float = 1.0):
        super().__init__()
        self.gate1 = nn.Linear(input_dim * 2, hidden_dim)
        self.gate2 = nn.Linear(hidden_dim, input_dim)
        self.gumbel = GumbelSoftmax(tau=tau, hard=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, C1_prime: torch.Tensor, C2_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate progressive mask and apply to features (Equations 14-16)

        Args:
            C1_prime, C2_prime: (batch_size, embed_dim)

        Returns:
            mask: (batch_size, embed_dim) binary mask
            C_masked: (batch_size, embed_dim) masked features
        """
        # Concatenate features
        C = torch.cat([C1_prime, C2_prime], dim=-1)  # (batch, 2*embed_dim)

        # Gating network (Equation 14)
        g = F.relu(self.gate1(C))
        g = self.sigmoid(self.gate2(g))  # (batch, embed_dim)

        # Reshape for binary sampling: (batch, embed_dim, 2)
        g_reshaped = g.unsqueeze(-1)  # (batch, embed_dim, 1)
        zeros = 1 - g_reshaped
        logits = torch.cat([g_reshaped, zeros], dim=-1)  # (batch, embed_dim, 2)

        # Gumbel-Softmax sampling (Equation 15)
        mask_soft = self.gumbel(logits)  # (batch, embed_dim, 2)
        mask = mask_soft[..., 0]  # (batch, embed_dim)

        # Apply mask (Equation 16)
        C_masked = C1_prime * mask + C2_prime * (1 - mask)

        return mask, C_masked


class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder (Equations 17-19)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.bigru = nn.GRU(
            input_dim, hidden_dim // 2, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            H: (batch_size, seq_len, hidden_dim)
        """
        # Add sequence dimension if input is 2D
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        outputs, _ = self.bigru(x)  # (batch, seq_len, hidden_dim)
        return outputs


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolutional Network layer (Equations 21-22)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch_size, num_nodes, in_dim)
            A_norm: (batch_size, num_nodes, num_nodes) normalized adjacency
        Returns:
            Z: (batch_size, num_nodes, out_dim)
        """
        # H @ W
        H_transformed = torch.matmul(H, self.weight)
        # A @ (H @ W)
        Z = torch.matmul(A_norm, H_transformed)
        return Z


class GraphConstructor(nn.Module):
    """
    Graph construction from features (Equation 20)
    """

    def __init__(self, k_neighbors: int = 5):
        super().__init__()
        self.k = k_neighbors

    def compute_cosine_similarity(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity matrix

        Args:
            H: (batch_size, num_nodes, hidden_dim)
        Returns:
            sim: (batch_size, num_nodes, num_nodes)
        """
        # Normalize for cosine similarity
        H_norm = F.normalize(H, p=2, dim=-1)
        sim = torch.matmul(H_norm, H_norm.transpose(-2, -1))
        return sim

    def build_knn_graph(self, H: torch.Tensor) -> torch.Tensor:
        """
        Build k-nearest neighbor graph (Equation 20)

        Args:
            H: (batch_size, num_nodes, hidden_dim)
        Returns:
            A: (batch_size, num_nodes, num_nodes) adjacency matrix
        """
        batch_size, num_nodes, _ = H.shape

        # Compute similarity
        sim = self.compute_cosine_similarity(H)  # (batch, num_nodes, num_nodes)

        # Get top-k neighbors
        _, topk_indices = torch.topk(sim, self.k, dim=-1)

        # Build adjacency matrix
        A = torch.zeros(batch_size, num_nodes, num_nodes, device=H.device)
        A.scatter_(-1, topk_indices, 1.0)

        # Add self-connections (Equation 21-22: A_tilde = A + I)
        A = A + torch.eye(num_nodes, device=H.device).unsqueeze(0)

        return A

    def normalize_adjacency(self, A: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix (Equations 21-22)
        D^{-1/2} * A * D^{-1/2}
        """
        # Compute degree matrix
        D = A.sum(dim=-1) + 1e-8  # (batch, num_nodes)
        D_inv_sqrt = D ** -0.5  # (batch, num_nodes)

        # D^{-1/2} * A * D^{-1/2}
        D_inv_sqrt = D_inv_sqrt.unsqueeze(-1)
        A_norm = D_inv_sqrt * A * D_inv_sqrt.transpose(-2, -1)

        return A_norm

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Build and normalize graph adjacency matrix

        Args:
            H: (batch_size, num_nodes, hidden_dim)
        Returns:
            A_norm: (batch_size, num_nodes, num_nodes) normalized adjacency
        """
        A = self.build_knn_graph(H)
        A_norm = self.normalize_adjacency(A)
        return A_norm


class HybridEncoder(nn.Module):
    """
    Hybrid encoder composed of BiGRU and GCN (Equations 17-22)
    """

    def __init__(self, input_dim: int, hidden_dim: int, gcn_hidden_dim: int = 128,
                 k_neighbors: int = 5, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # BiGRU encoder
        self.bigru = BiGRUEncoder(hidden_dim, hidden_dim, dropout=dropout)

        # Graph constructor
        self.graph_constructor = GraphConstructor(k_neighbors=k_neighbors)

        # GCN layers (Equations 21-22)
        self.gcn1 = GraphConvolutionLayer(hidden_dim, gcn_hidden_dim)
        self.gcn2 = GraphConvolutionLayer(gcn_hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, C_masked: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C_masked: (batch_size, embed_dim)
        Returns:
            Z_encoded: (batch_size, seq_len, hidden_dim)
        """
        # Determine sequence length
        batch_size = C_masked.size(0)
        embed_dim = C_masked.size(1)
        seq_len = int(math.sqrt(embed_dim))
        hidden_dim = embed_dim // seq_len

        # Reshape to sequence (Equation 17)
        C_seq = C_masked.view(batch_size, seq_len, hidden_dim)
        C_proj = self.input_proj(C_seq)  # (batch, seq_len, hidden_dim)

        # BiGRU encoding (Equations 17-19)
        H = self.bigru(C_proj)  # (batch, seq_len, hidden_dim)

        # Build graph (Equation 20)
        A_norm = self.graph_constructor(H)  # (batch, seq_len, seq_len)

        # GCN message passing (Equation 21)
        Z1 = self.gcn1(H, A_norm)
        Z1 = self.relu(Z1)
        Z1 = self.dropout(Z1)

        # Second GCN layer (Equation 22)
        Z_encoded = self.gcn2(Z1, A_norm)

        return Z_encoded


class ReverseGRUDecoder(nn.Module):
    """
    Reverse GRU decoder (Equation 23-24)
    """

    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, Z_encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_encoded: (batch_size, seq_len, hidden_dim)
        Returns:
            D: (batch_size, seq_len, hidden_dim) reverse decoded features
        """
        # Reverse sequence order (Equation 23)
        Z_reversed = torch.flip(Z_encoded, dims=[1])

        # GRU decoding
        D, _ = self.gru(Z_reversed)  # (batch, seq_len, hidden_dim)

        return D


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer (Equations 25-27)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.key_proj = nn.Linear(hidden_dim, attention_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Cross-modal attention (Equations 25-27)

        Args:
            C: (batch_size, embed_dim) original features
            D: (batch_size, seq_len, hidden_dim) decoded features
        Returns:
            C_attn: (batch_size, hidden_dim) weighted context vector
        """
        batch_size, seq_len, hidden_dim = D.shape

        # Project original features to match sequence length (Equation 26)
        # Expand C to (batch, seq_len, hidden_dim)
        if C.dim() == 2:
            C_expanded = C.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            C_expanded = C

        # Compute attention scores (Equation 26)
        Q = self.query_proj(C_expanded)  # (batch, seq_len, attn_dim)
        K = self.key_proj(D)  # (batch, seq_len, attn_dim)
        V = self.value_proj(D)  # (batch, seq_len, hidden_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)

        # Attention weights (Equation 25)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Apply attention to values (Equation 27)
        C_attn = torch.matmul(attn_weights, V)  # (batch, seq_len, hidden_dim)
        C_attn = C_attn.mean(dim=1)  # (batch, hidden_dim)

        return C_attn


class HybridDecoder(nn.Module):
    """
    Hybrid decoder with reverse GRU and cross-modal attention (Equations 23-28)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.reverse_decoder = ReverseGRUDecoder(hidden_dim, dropout=dropout)
        self.cross_attention = CrossModalAttention(hidden_dim, attention_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, Z_encoded: torch.Tensor, C_original: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_encoded: (batch_size, seq_len, hidden_dim) from encoder
            C_original: (batch_size, embed_dim) original features
        Returns:
            C_decoded: (batch_size, hidden_dim)
        """
        # Reverse GRU decoding (Equations 23-24)
        D = self.reverse_decoder(Z_encoded)  # (batch, seq_len, hidden_dim)

        # Cross-modal attention (Equations 25-27)
        C_attn = self.cross_attention(C_original, D)

        # Residual connection and output (Equation 28)
        # Project C_original to match dimension
        if C_original.size(1) != Z_encoded.size(2):
            C_proj = C_original[:, :Z_encoded.size(2)]
        else:
            C_proj = C_original

        C_decoded = self.relu(self.output_proj(C_attn + C_proj))

        return C_decoded


class MaskEnhancedClassifier(nn.Module):
    """
    Complete Mask Enhanced Classifier (Section 3.4.3)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Progressive learnable mask (Equations 14-16)
        self.progressive_mask = ProgressiveLearnableMask(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            tau=config.gumbel_tau
        )

        # Hybrid encoder (Equations 17-22)
        self.encoder = HybridEncoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            gcn_hidden_dim=config.gcn_hidden_dim,
            k_neighbors=config.k_neighbors,
            dropout=config.dropout
        )

        # Hybrid decoder (Equations 23-28)
        self.decoder = HybridDecoder(
            hidden_dim=config.hidden_dim,
            attention_dim=config.hidden_dim // 4,
            dropout=config.dropout
        )

        # Classifier
        self.classifier = nn.Linear(config.hidden_dim, 2)

        # Temperature annealing
        self.tau = config.gumbel_tau
        self.tau_min = config.gumbel_tau_min
        self.anneal_rate = config.gumbel_anneal_rate

    def update_temperature(self, epoch: int, total_epochs: int):
        """
        Anneal temperature for Gumbel-Softmax
        """
        self.tau = max(self.tau_min, self.tau * self.anneal_rate)
        self.progressive_mask.gumbel.tau = self.tau

    def forward(self, C1_prime: torch.Tensor, C2_prime: torch.Tensor,
                C_original: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            C1_prime, C2_prime: (batch_size, hidden_dim)
            C_original: (batch_size, hidden_dim * 2) or (batch_size, hidden_dim)
        Returns:
            logits: (batch_size, 2)
            C_decoded: (batch_size, hidden_dim)
        """
        # Progressive learnable masking (Equations 14-16)
        mask, C_masked = self.progressive_mask(C1_prime, C2_prime)

        # Hybrid encoding (Equations 17-22)
        Z_encoded = self.encoder(C_masked)

        # Hybrid decoding (Equations 23-28)
        C_decoded = self.decoder(Z_encoded, C_original)

        # Classification
        logits = self.classifier(C_decoded)

        return logits, C_decoded