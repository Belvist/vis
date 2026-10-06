"""Adaptive token pooling - TokenLearner style for variable token count"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class TokenLearner(nn.Module):
    """
    TokenLearner: learns to pool spatial features into a small number of adaptive tokens.
    From "TokenLearner: Adaptive Space-Time Tokenization for Videos" (Ryoo et al., 2021)
    """
    
    def __init__(self, 
                 in_channels: int,
                 num_tokens: int = 16,
                 bottleneck_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_channels = in_channels
        
        # Attention network: spatial -> token weights
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, 1),
            nn.GroupNorm(4, bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, num_tokens, 1),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Optional: token refinement
        self.refine = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] spatial features
        Returns: [B, num_tokens, C] tokens
        """
        B, C, H, W = x.shape
        
        # Compute attention maps for each token
        attn_maps = self.attention(x)  # [B, num_tokens, H, W]
        attn_maps = attn_maps.view(B, self.num_tokens, -1)  # [B, num_tokens, H*W]
        attn_weights = F.softmax(attn_maps, dim=-1)  # [B, num_tokens, H*W]
        
        # Weighted pooling
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        tokens = torch.bmm(attn_weights, x_flat.transpose(1, 2))  # [B, num_tokens, C]
        
        # Optional refinement
        tokens = self.refine(tokens)
        
        return tokens


class MultiScaleTokenLearner(nn.Module):
    """
    TokenLearner applied to multiple feature scales.
    Combines tokens from F4, F8, F16, F32 levels.
    """
    
    def __init__(self, 
                 channels_list: List[int],
                 num_tokens_per_scale: List[int],
                 bottleneck_dim: int = 64):
        super().__init__()
        assert len(channels_list) == len(num_tokens_per_scale)
        
        self.token_learners = nn.ModuleList([
            TokenLearner(ch, n_tokens, bottleneck_dim)
            for ch, n_tokens in zip(channels_list, num_tokens_per_scale)
        ])
        
        self.total_tokens = sum(num_tokens_per_scale)
        self.scales = len(channels_list)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: List of [B, C_i, H_i, W_i] for each scale
        Returns: [B, total_tokens, C] where C is projected to common dim
        """
        all_tokens = []
        for feat, learner in zip(features, self.token_learners):
            tokens = learner(feat)  # [B, n_tokens, C_i]
            all_tokens.append(tokens)
        
        # Concatenate tokens from all scales
        tokens = torch.cat(all_tokens, dim=1)  # [B, total_tokens, C_varies]
        
        return tokens


class AdaptiveTokenPooler(nn.Module):
    """
    Simpler adaptive pooling: uses learned queries to attend to spatial features.
    More like Perceiver IO / cross-attention pooling.
    """
    
    def __init__(self, 
                 in_channels: int,
                 num_tokens: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Learnable token queries
        self.token_queries = nn.Parameter(torch.randn(num_tokens, in_channels) * 0.02)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] spatial features
        Returns: [B, num_tokens, C] tokens
        """
        B, C, H, W = x.shape
        
        # Flatten spatial
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Expand queries to batch
        queries = self.token_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, C]
        
        # Cross-attention: tokens attend to spatial features
        tokens, _ = self.cross_attn(
            query=self.norm1(queries),
            key=self.norm1(x_flat),
            value=x_flat
        )
        tokens = tokens + queries  # residual
        
        # FFN
        tokens = tokens + self.ffn(self.norm2(tokens))
        
        return tokens


def create_adaptive_pooler(config: dict) -> nn.Module:
    """Factory for adaptive token pooler"""
    pooler_type = config.get("type", "tokenlearner")
    
    if pooler_type == "tokenlearner":
        return TokenLearner(
            in_channels=config["in_channels"],
            num_tokens=config.get("num_tokens", 16),
            bottleneck_dim=config.get("bottleneck_dim", 64),
            dropout=config.get("dropout", 0.0)
        )
    elif pooler_type == "multiscale_tokenlearner":
        return MultiScaleTokenLearner(
            channels_list=config["channels_list"],
            num_tokens_per_scale=config.get("num_tokens_per_scale", [4, 4, 4, 4]),
            bottleneck_dim=config.get("bottleneck_dim", 64)
        )
    elif pooler_type == "cross_attention":
        return AdaptiveTokenPooler(
            in_channels=config["in_channels"],
            num_tokens=config.get("num_tokens", 16),
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")