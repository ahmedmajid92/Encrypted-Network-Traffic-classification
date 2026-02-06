"""
Rotary FT-Transformer for Tabular Classification
=================================================
Main model class combining FeatureTokenizer, Transformer Encoder, and Classification Head.

PhD Research: Encrypted Traffic Classification with Protocol Hierarchy Encoding
"""

import torch
import torch.nn as nn

from .components import FeatureTokenizer, TransformerEncoderLayer


class RotaryFTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer with Rotary Position Embeddings.
    
    Novel contribution: Using RoPE to encode protocol depth hierarchy
    where position represents IP -> TCP/UDP -> derived features.
    """
    
    def __init__(
        self,
        cat_cardinalities: dict[str, int],
        num_features: list[str],
        num_classes: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            cat_cardinalities: Dict mapping categorical feature names to their cardinalities
            num_features: List of numerical feature names
            num_classes: Number of output classes (2 for binary VPN classification)
            embed_dim: Token embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            ff_dim: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_features = len(cat_cardinalities) + len(num_features)
        self.embed_dim = embed_dim
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            embed_dim=embed_dim
        )
        
        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                max_seq_len=self.num_features + 1  # +1 for CLS
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: (Batch, Num_Cat_Features) - categorical indices
            x_num: (Batch, Num_Num_Features) - numerical values
            
        Returns:
            logits: (Batch, Num_Classes)
        """
        B = x_cat.size(0)
        
        # Tokenize features: (B, N, D)
        x = self.tokenizer(x_cat, x_num)
        
        # Prepend CLS token: (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Pass through transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Extract CLS token representation
        cls_output = x[:, 0]  # (B, D)
        
        # Classification
        logits = self.classifier(cls_output)  # (B, num_classes)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    cat_cardinalities: dict[str, int],
    num_features: list[str],
    num_classes: int = 2,
    config: dict = None
) -> RotaryFTTransformer:
    """
    Factory function to create a RotaryFTTransformer with optional config.
    
    Args:
        cat_cardinalities: Dict of categorical feature cardinalities
        num_features: List of numerical feature names
        num_classes: Number of output classes
        config: Optional dict with model hyperparameters
        
    Returns:
        RotaryFTTransformer instance
    """
    default_config = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "ff_dim": 256,
        "dropout": 0.1
    }
    
    if config:
        default_config.update(config)
    
    return RotaryFTTransformer(
        cat_cardinalities=cat_cardinalities,
        num_features=num_features,
        num_classes=num_classes,
        **default_config
    )
