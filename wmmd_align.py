"""
WMMD-Align: Weighted Maximum Mean Discrepancy Alignment (Section 3.3)
Equations 6-8 and Equation 31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GaussianKernel(nn.Module):
    """
    Gaussian kernel function (Equation 7)
    k(t, v) = exp(-||t - v||^2 / (2 * sigma^2))
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t, v: (batch_size, feature_dim)
        Returns:
            kernel: (batch_size,)
        """
        diff = t - v
        squared_dist = torch.sum(diff ** 2, dim=1)
        return torch.exp(-squared_dist / (2 * self.sigma ** 2))


class WeightedMMD(nn.Module):
    """
    Weighted Maximum Mean Discrepancy
    Combines MMD with learnable weighted Frobenius norm (Equation 6)
    """

    def __init__(self, feature_dim: int, sigma: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.kernel = GaussianKernel(sigma)

        # Learnable weight matrix W_W (Equation 6)
        self.weight_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)

    def compute_mmd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute standard MMD between source and target distributions

        MMD^2 = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
        """
        batch_size = source.size(0)

        # Compute kernel matrices
        # For efficiency, use unbiased estimator with random subsets
        # Full MMD computation would be O(n^2), here we use a simplified version

        # Sample indices for efficiency (if batch is large)
        if batch_size > 100:
            indices = torch.randperm(batch_size)[:100]
            source_sample = source[indices]
            target_sample = target[indices]
        else:
            source_sample = source
            target_sample = target

        # Compute kernel similarities
        k_ss = 0
        k_tt = 0
        k_st = 0

        n = source_sample.size(0)

        for i in range(n):
            for j in range(i, n):
                k_ss += self.kernel(source_sample[i], source_sample[j])
                k_tt += self.kernel(target_sample[i], target_sample[j])
                k_st += self.kernel(source_sample[i], target_sample[j])

        # Normalize
        k_ss = 2 * k_ss / (n * n)
        k_tt = 2 * k_tt / (n * n)
        k_st = 2 * k_st / (n * n)

        mmd = k_ss + k_tt - 2 * k_st
        return mmd

    def compute_weighted_frobenius(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted Frobenius norm (Equation 6 second term)
        ||W_W * (source - target)||_F
        """
        diff = source - target  # (batch, feature_dim)

        # Apply learnable weight matrix
        # (batch, feature_dim) @ (feature_dim, feature_dim) -> (batch, feature_dim)
        weighted_diff = torch.matmul(diff, self.weight_matrix)

        # Compute Frobenius norm (sum of squares)
        frobenius_norm = torch.sum(weighted_diff ** 2)

        return frobenius_norm

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute WMMD-Align loss (Equation 6)

        Args:
            source: (batch_size, feature_dim) source domain features
            target: (batch_size, feature_dim) target domain features

        Returns:
            loss: scalar
        """
        mmd_loss = self.compute_mmd(source, target)
        frob_loss = self.compute_weighted_frobenius(source, target)

        return mmd_loss + frob_loss


class DomainAlignmentLoss(nn.Module):
    """
    Domain alignment loss combining three WMMD-Align terms (Equation 8)
    loss_WMMD-Align = α * WMMD(M_text, M_img) + β * WMMD(M_text, M_ded) + γ * WMMD(M_text, M_red)
    """

    def __init__(self, feature_dim: int, sigma: float = 1.0,
                 alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        super().__init__()
        self.wmmd = WeightedMMD(feature_dim, sigma)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, M_text: torch.Tensor, M_img: torch.Tensor,
                M_ded: torch.Tensor, M_red: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M_text: (batch_size, feature_dim) text features
            M_img: (batch_size, feature_dim) image features
            M_ded: (batch_size, feature_dim) DED features
            M_red: (batch_size, feature_dim) RED features

        Returns:
            loss: scalar (Equation 8)
        """
        loss_img = self.wmmd(M_text, M_img)
        loss_ded = self.wmmd(M_text, M_ded)
        loss_red = self.wmmd(M_text, M_red)

        total_loss = self.alpha * loss_img + self.beta * loss_ded + self.gamma * loss_red

        return total_loss


class AdaptiveDomainAlignmentLoss(nn.Module):
    """
    Adaptive domain alignment loss with learnable fusion parameters
    """

    def __init__(self, feature_dim: int, sigma: float = 1.0):
        super().__init__()
        self.wmmd = WeightedMMD(feature_dim, sigma)

        # Learnable fusion parameters (alpha, beta, gamma)
        self.logits = nn.Parameter(torch.ones(3) / 3)

    def forward(self, M_text: torch.Tensor, M_img: torch.Tensor,
                M_ded: torch.Tensor, M_red: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive domain alignment loss
        """
        loss_img = self.wmmd(M_text, M_img)
        loss_ded = self.wmmd(M_text, M_ded)
        loss_red = self.wmmd(M_text, M_red)

        # Compute adaptive weights using softmax
        weights = torch.softmax(self.logits, dim=0)
        alpha, beta, gamma = weights[0], weights[1], weights[2]

        total_loss = alpha * loss_img + beta * loss_ded + gamma * loss_red

        return total_loss