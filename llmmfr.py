"""
Complete LLMMFR model integrating all modules
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from models.feature_extractor import FeatureExtractor
from models.mccnn import MultiChannelFeatureExtractor
from models.wmmd_align import DomainAlignmentLoss
from models.dual_attention import MainClassifier
from models.mask_enhanced_classifier import MaskEnhancedClassifier
from models.losses import TotalLoss


class LLMMFR(nn.Module):
    """
    Large Language Model Enhanced Fake News Detection with Masked Feature Reconstruction
    Complete implementation of the LLMMFR framework
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature extraction module (Section 3.1)
        self.feature_extractor = FeatureExtractor(config)

        # Multi-channel feature extraction (Section 3.3)
        self.multi_channel_extractor = MultiChannelFeatureExtractor(config)

        # Domain alignment loss (Section 3.3)
        self.domain_alignment = DomainAlignmentLoss(
            feature_dim=config.hidden_dim,
            sigma=config.sigma,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma
        )

        # Main classifier with dual-attention (Sections 3.4.1-3.4.2)
        self.main_classifier = MainClassifier(config)

        # Mask Enhanced Classifier (Section 3.4.3)
        self.mask_classifier = MaskEnhancedClassifier(config)

        # Final classification layer (Equation 29)
        final_input_dim = config.hidden_dim * 2  # Concatenation of C_original and C_decoded
        self.final_classifier = nn.Sequential(
            nn.Linear(final_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2)
        )

        # Loss function
        self.loss_fn = TotalLoss(
            lambda_wmmd=config.lambda_wmmd,
            use_mask_classifier=True,
            mask_weight=0.5
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                images: torch.Tensor, ded_input_ids: torch.Tensor,
                ded_attention_mask: torch.Tensor, red_input_ids: torch.Tensor,
                red_attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LLMMFR

        Args:
            input_ids, attention_mask: for news document
            images: for news image
            ded_input_ids, ded_attention_mask: for DED
            red_input_ids, red_attention_mask: for RED
            labels: optional ground truth labels for loss computation

        Returns:
            output: dictionary containing logits, losses, and intermediate outputs
        """

        # ==================== Section 3.1: Feature vector generation ====================
        # Encode news document (Equation 1)
        DV_text = self.feature_extractor.encode_text(input_ids, attention_mask)

        # Encode news image (Equation 2)
        PV_img = self.feature_extractor.encode_image(images)

        # Encode DED (Equation 3)
        PV_ded = self.feature_extractor.encode_enhanced_document(ded_input_ids, ded_attention_mask)

        # Encode RED (Equation 4)
        PV_red = self.feature_extractor.encode_enhanced_document(red_input_ids, red_attention_mask)

        # ==================== Section 3.3: Multi-channel feature extraction ====================
        # Extract multi-view representations (Equations 5-8)
        M_text, M_img, M_ded, M_red = self.multi_channel_extractor(DV_text, PV_img, PV_ded, PV_red)

        # ==================== Section 3.3: Domain alignment loss ====================
        loss_domain = self.domain_alignment(M_text, M_img, M_ded, M_red)

        # ==================== Sections 3.4.1-3.4.2: Main classifier ====================
        C1, C2, C1_prime, C2_prime = self.main_classifier(
            M_text, M_img, M_ded, M_red, DV_text, PV_img
        )

        # Concatenate C1 and C2 for final classification (Equation 29)
        C_original = torch.cat([C1, C2], dim=-1)  # (batch, hidden_dim * 2)

        # ==================== Section 3.4.3: Mask Enhanced Classifier ====================
        logits_mask, C_decoded = self.mask_classifier(C1_prime, C2_prime, C_original)

        # ==================== Section 3.4.4: Final classification ====================
        # Concatenate C_original and C_decoded (Equation 29)
        C_final = torch.cat([C_original, C_decoded], dim=-1)
        logits_final = self.final_classifier(C_final)

        # ==================== Output ====================
        output = {
            'logits_final': logits_final,
            'logits_mask': logits_mask,
            'loss_domain': loss_domain,
            'C1': C1,
            'C2': C2,
            'C1_prime': C1_prime,
            'C2_prime': C2_prime,
            'C_decoded': C_decoded
        }

        # Compute losses if labels are provided
        if labels is not None:
            total_loss, loss_mul, loss_domain_val = self.loss_fn(
                logits_final, logits_mask, labels, loss_domain
            )
            output['total_loss'] = total_loss
            output['loss_mul'] = loss_mul
            output['loss_domain'] = loss_domain_val

        return output

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                images: torch.Tensor, ded_input_ids: torch.Tensor,
                ded_attention_mask: torch.Tensor, red_input_ids: torch.Tensor,
                red_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Prediction function for inference

        Returns:
            predictions: (batch_size,) predicted class labels
        """
        with torch.no_grad():
            output = self.forward(
                input_ids, attention_mask, images,
                ded_input_ids, ded_attention_mask,
                red_input_ids, red_attention_mask,
                labels=None
            )
            predictions = torch.argmax(output['logits_final'], dim=-1)
        return predictions

    def update_temperature(self, epoch: int, total_epochs: int):
        """
        Update Gumbel-Softmax temperature during training
        """
        self.mask_classifier.update_temperature(epoch, total_epochs)