"""
Loss functions for LLMMFR
Sections 3.4.4 and 3.5
Equations 29-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for classification (Equation 30)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        Returns:
            loss: scalar
        """
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        return loss


class MFNDLoss(nn.Module):
    """
    Multimodal Fake News Detection Loss (Equation 30)
    loss_mul = -∑∑(y_ij * log(ŷ_ij) + (1 - y_ij) * log(1 - ŷ_ij))
    """

    def __init__(self, use_mask_classifier: bool = True, mask_weight: float = 0.5):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.use_mask_classifier = use_mask_classifier
        self.mask_weight = mask_weight

    def forward(self, logits_final: torch.Tensor, logits_mask: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_final: (batch_size, 2) from final classifier (Equation 29)
            logits_mask: (batch_size, 2) from mask classifier
            labels: (batch_size,) ground truth labels

        Returns:
            loss_main: cross-entropy loss for final classifier
            loss_mask: cross-entropy loss for mask classifier
            loss_mul: combined MFND loss (Equation 30)
        """
        loss_main = self.ce_loss(logits_final, labels)
        loss_mask = self.ce_loss(logits_mask, labels)

        # Combined loss (Equation 30)
        if self.use_mask_classifier:
            loss_mul = loss_main + self.mask_weight * loss_mask
        else:
            loss_mul = loss_main

        return loss_main, loss_mask, loss_mul


class TotalLoss(nn.Module):
    """
    Total model loss combining domain alignment loss and MFND loss (Equation 31)
    loss = loss_mul + λ * loss_WMMD-Align
    """

    def __init__(self, lambda_wmmd: float = 0.7, use_mask_classifier: bool = True, mask_weight: float = 0.5):
        super().__init__()
        self.lambda_wmmd = lambda_wmmd
        self.mfnd_loss = MFNDLoss(use_mask_classifier, mask_weight)

    def forward(self, logits_final: torch.Tensor, logits_mask: torch.Tensor,
                labels: torch.Tensor, loss_domain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits_final: (batch_size, 2)
            logits_mask: (batch_size, 2)
            labels: (batch_size,)
            loss_domain: scalar domain alignment loss

        Returns:
            total_loss: scalar (Equation 31)
            loss_mul: scalar MFND loss
            loss_domain: scalar domain alignment loss
        """
        loss_main, loss_mask, loss_mul = self.mfnd_loss(logits_final, logits_mask, labels)

        # Total loss (Equation 31)
        total_loss = loss_mul + self.lambda_wmmd * loss_domain

        return total_loss, loss_mul, loss_domain


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification (optional alternative to cross-entropy)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        Returns:
            loss: scalar
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()