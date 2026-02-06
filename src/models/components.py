"""
Rotary FT-Transformer Components
================================
Core building blocks: FeatureTokenizer, RoPE, ReGLU, RotaryMultiHeadAttention

PhD Research: Encrypted Traffic Classification with Protocol Hierarchy Encoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class FeatureTokenizer(nn.Module):
    """
    Converts tabular inputs (categorical + numerical) into a sequence of embeddings.
    
    Output shape: (Batch, Num_Features, Embed_Dim)
    """
    
    def __init__(
        self,
        cat_cardinalities: dict[str, int],  # {feature_name: num_unique_values}
        num_features: list[str],             # List of numerical feature names
        embed_dim: int = 64
    ):
        super().__init__()
        self.cat_names = list(cat_cardinalities.keys())
        self.num_names = num_features
        self.embed_dim = embed_dim
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinality + 1, embed_dim, padding_idx=0)  # +1 for unknown
            for name, cardinality in cat_cardinalities.items()
        })
        
        # Numerical projections (scalar -> embed_dim)
        self.num_projections = nn.ModuleDict({
            name: nn.Linear(1, embed_dim)
            for name in num_features
        })
        
        # Layer norm for each feature
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: (Batch, Num_Cat_Features) - integer indices
            x_num: (Batch, Num_Num_Features) - float values
            
        Returns:
            (Batch, Num_Total_Features, Embed_Dim)
        """
        tokens = []
        
        # Embed categorical features
        for i, name in enumerate(self.cat_names):
            emb = self.cat_embeddings[name](x_cat[:, i])  # (B, D)
            tokens.append(emb)
        
        # Project numerical features
        for i, name in enumerate(self.num_names):
            proj = self.num_projections[name](x_num[:, i:i+1])  # (B, D)
            tokens.append(proj)
        
        # Stack into sequence: (B, N, D)
        x = torch.stack(tokens, dim=1)
        x = self.layer_norm(x)
        
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) - Su et al. 2021
    
    Applies rotation to Q and K vectors based on position.
    Unlike NLP, here position = Protocol Depth (IP -> TCP/UDP -> derived).
    """
    
    def __init__(self, dim: int, max_seq_len: int = 64, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute sin/cos for all positions
        self._build_cache(max_seq_len)
        
    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin for the given sequence length.
        """
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary embeddings to Q and K tensors.
    
    Args:
        q, k: (Batch, Heads, SeqLen, HeadDim)
        cos, sin: (SeqLen, HeadDim)
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    # Expand cos/sin to match batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, SeqLen, HeadDim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class ReGLU(nn.Module):
    """
    ReGLU activation: ReLU(xW1) ⊙ xW2
    
    Proven better than ReLU/GELU for tabular transformers.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.w1(x))
        value = self.w2(x)
        x = gate * value
        x = self.dropout(x)
        x = self.w3(x)
        return x


class RotaryMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embeddings.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 64):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, EmbedDim)
        Returns:
            (Batch, SeqLen, EmbedDim)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (B, H, N, HeadDim)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        
        # Apply RoPE to Q and K
        cos, sin = self.rope(x, N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, N, HeadDim)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        
        return out


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder layer with RoPE attention and ReGLU FFN.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RotaryMultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = ReGLU(embed_dim, ff_dim, embed_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
