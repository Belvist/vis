"""Minimal Gate 1 Student - only backbone + token_pooler"""
import torch
import torch.nn as nn
from typing import Optional

from earai.models.backbone import create_backbone
from earai.heads.adaptive_tokens import create_adaptive_pooler
from earai.core.config import EarAIConfig, DEFAULT_CONFIG


class Gate1Student(nn.Module):
    """
    Minimal Gate 1 Student - only backbone + token_pooler.
    No residual encoder, state updater, scene memory, etc.
    """
    
    def __init__(self, config: EarAIConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        
        # Core components - only backbone and token pooler
        self.backbone = create_backbone(self.config)
        
        # Adaptive token pooler
        self.token_pooler = create_adaptive_pooler({
            "type": "multiscale_tokenlearner",
            "channels_list": [self.config.feature_dim] * 4,
            "num_tokens_per_scale": [4, 4, 4, 4],  # 16 total
            "bottleneck_dim": 64
        })
        
        # Initialize state
        self.register_buffer('prev_frame', torch.zeros(1, 3, 224, 224))
        self.frame_id = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass for a single frame (keyframe path).
        Returns visual tokens: [B, 16, 256]
        """
        B = x.shape[0]
        
        # Full backbone
        backbone_out = self.backbone(x)  # Dict with F4, F8, F16, F32
        
        # Extract multi-scale features in correct order for TokenLearner
        features = [
            backbone_out["F4"],
            backbone_out["F8"],
            backbone_out["F16"],
            backbone_out["F32"],
        ]
        
        # Token learner
        token_result = self.token_pooler(features, return_attention=False)
        # Handle both dict and tensor returns
        if isinstance(token_result, dict):
            tokens = token_result['tokens']
        else:
            tokens = token_result  # [B, 16, 256]
        
        return tokens
    
    def reset(self):
        """Reset streaming state"""
        self.prev_frame.zero_()
        self.frame_id = 0


def create_gate1_student(config: dict) -> Gate1Student:
    """Factory for minimal Gate 1 student"""
    earai_config = EarAIConfig(
        backbone_pretrained=config.get('backbone_pretrained', True),
        feature_dim=config.get('feature_dim', 256),
        num_scene_tokens=config.get('num_scene_tokens', 16),
        peripheral_resolution=config.get('peripheral_resolution', (224, 224)),
        fovea_resolution=config.get('fovea_resolution', (600, 600)),
        max_entities=config.get('max_entities', 100)
    )
    
    return Gate1Student(earai_config)