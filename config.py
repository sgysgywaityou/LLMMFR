"""
Configuration file for LLMMFR model
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Config:
    """Configuration class for LLMMFR model"""

    # ==================== Data settings ====================
    dataset_name: str = "GossipCop"  # Options: GossipCop, Weibo, PolitiFact
    data_root: str = "./data"
    max_text_length: int = 200
    image_size: Tuple[int, int] = (224, 224)

    # ==================== Feature dimensions ====================
    text_dim: int = 768          # RoBERTa output dimension
    image_dim: int = 768         # MAE output dimension (after projection)
    hidden_dim: int = 512        # Hidden dimension for attention and GCN
    num_heads: int = 8           # Number of attention heads
    gcn_hidden_dim: int = 256    # GCN hidden dimension

    # ==================== MCCNN settings (Section 3.3) ====================
    num_text_channels: int = 10
    num_image_channels: int = 10
    num_ded_channels: int = 10
    num_red_channels: int = 10
    conv_kernel_size: int = 3

    # ==================== WMMD-Align parameters (Section 3.3) ====================
    sigma: float = 1.0           # Gaussian kernel bandwidth
    alpha: float = 0.4           # Fusion weight for image alignment
    beta: float = 0.3            # Fusion weight for DED alignment
    gamma: float = 0.3           # Fusion weight for RED alignment

    # ==================== Mask Enhanced Classifier (Section 3.4.3) ====================
    gumbel_tau: float = 1.0      # Initial temperature for Gumbel-Softmax
    gumbel_tau_min: float = 0.5  # Minimum temperature
    gumbel_anneal_rate: float = 0.99  # Temperature annealing rate
    k_neighbors: int = 5         # K for k-NN graph
    gcn_layers: int = 2          # Number of GCN layers

    # ==================== Training settings ====================
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.5
    lambda_wmmd: float = 0.7     # Domain alignment loss weight (Equation 31)
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # ==================== LLM settings (Section 3.2) ====================
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")

    # ==================== Paths ====================
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # ==================== Device ====================
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Default configuration
default_config = Config()