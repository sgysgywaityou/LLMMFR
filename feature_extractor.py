"""
Feature vector generation module (Section 3.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import timm
from typing import Tuple, Optional


class RoBERTaEncoder(nn.Module):
    """
    RoBERTa-based text encoder (Equation 1)
    """

    def __init__(self, model_name: str = "roberta-base", output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        # Projection layer to match target dimension if needed
        if self.roberta.config.hidden_size != output_dim:
            self.projection = nn.Linear(self.roberta.config.hidden_size, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            text_features: (batch_size, seq_len, output_dim)
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use the last hidden states (token-level representations)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, roberta_dim)
        token_embeddings = self.dropout(token_embeddings)

        # Project to output dimension
        text_features = self.projection(token_embeddings)

        return text_features


class MAEEncoder(nn.Module):
    """
    MAE (Masked Autoencoder) based image encoder (Equation 2)
    """

    def __init__(self, model_name: str = "mae_vit_base_patch16", output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        # Load pretrained MAE
        self.mae = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        # Get the dimension from the model
        if hasattr(self.mae, 'num_features'):
            mae_dim = self.mae.num_features
        else:
            mae_dim = 768

        # Projection layer
        if mae_dim != output_dim:
            self.projection = nn.Linear(mae_dim, output_dim)
        else:
            self.projection = nn.Identity()

        # Number of patches for ViT base with 224x224 input and 16x16 patches
        self.num_patches = 196  # (224/16)^2 = 196

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, 224, 224)

        Returns:
            image_features: (batch_size, num_patches, output_dim)
        """
        # Extract patch embeddings from MAE encoder
        # forward_features returns patch embeddings for ViT-based models
        if hasattr(self.mae, 'forward_features'):
            patch_embeddings = self.mae.forward_features(images)  # (batch, num_patches, mae_dim)
        else:
            # Fallback: use the model's forward and extract features
            # This is a simplified approach
            with torch.no_grad():
                features = self.mae(images)  # (batch, mae_dim)
            # Expand to patch-level representation (simplified)
            patch_embeddings = features.unsqueeze(1).expand(-1, self.num_patches, -1)

        patch_embeddings = self.dropout(patch_embeddings)
        image_features = self.projection(patch_embeddings)

        return image_features


class FeatureProjector(nn.Module):
    """
    Project features from different modalities to a common space
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class FeatureExtractor(nn.Module):
    """
    Combined feature extractor for all modalities
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text encoder for news documents
        self.text_encoder = RoBERTaEncoder(
            model_name="roberta-base",
            output_dim=config.text_dim,
            dropout=config.dropout
        )

        # Image encoder for news images
        self.image_encoder = MAEEncoder(
            model_name="mae_vit_base_patch16",
            output_dim=config.image_dim,
            dropout=config.dropout
        )

        # Note: DED and RED are also encoded using the same RoBERTa encoder
        # but this will be done separately in the LLM enhancement module

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode news document"""
        return self.text_encoder(input_ids, attention_mask)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode news image"""
        return self.image_encoder(images)

    def encode_enhanced_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode DED or RED using the same RoBERTa encoder (Equations 3-4)"""
        return self.text_encoder(input_ids, attention_mask)