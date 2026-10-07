"""Gate 2 Student - Minimal UI Student Model"""
import torch
import torch.nn as nn
from typing import Optional

from typing import Dict, Optional

from earai.core.config import EarAIConfig, DEFAULT_CONFIG
from earai.models.backbone import create_backbone
from earai.heads.adaptive_tokens import create_adaptive_pooler
from earai.training.gate2_decoder import Gate2Decoder


class Gate2Student(nn.Module):
    """
    Gate 2 Student - Minimal UI Student Model.
    Only backbone + token pooler + UI decoder.
    """
    
    def __init__(self, config: EarAIConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        
        # Core components - only backbone + token pooler
        self.backbone = create_backbone(self.config)
        
        # Adaptive token pooler
        self.token_pooler = create_adaptive_pooler({
            "type": "multiscale_tokenlearner",
            "channels_list": [self.config.feature_dim] * 4,
            "num_tokens_per_scale": [4, 4, 4, 4],  # 16 total
            "bottleneck_dim": 64
        })
        
        # UI Decoder
        self.decoder = Gate2Decoder(
            token_dim=self.config.feature_dim,
            num_classes=15,
            num_queries=32,
            hidden_dim=256,
            style_dim=10
        )
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Full forward pass.
        Returns decoder outputs with all UI predictions AND tokens for loss computation.
        """
        B = x.shape[0]
        
        # Full backbone
        backbone_out = self.backbone(x)
        
        # Extract multi-scale features
        features = [
            backbone_out["F4"],
            backbone_out["F8"],
            backbone_out["F16"],
            backbone_out["F32"],
        ]
        
        # Token learner
        token_result = self.token_pooler(features, return_attention=False)
        if isinstance(token_result, dict):
            tokens = token_result['tokens']
        else:
            tokens = token_result  # [B, 16, 256]
        
        # UI Decoder
        ui_out = self.decoder(tokens)
        
        # Return both tokens and decoder outputs for loss computation
        return {
            'tokens': tokens,
            **ui_out
        }


def create_gate2_student(config: dict) -> nn.Module:
    """Factory for Gate 2 student"""
    earai_config = EarAIConfig(
        backbone_pretrained=config.get('backbone_pretrained', True),
        feature_dim=config.get('feature_dim', 256),
        num_scene_tokens=config.get('num_scene_tokens', 16),
        peripheral_resolution=config.get('peripheral_resolution', (224, 224)),
        fovea_resolution=config.get('fovea_resolution', (600, 600)),
        max_entities=config.get('max_entities', 50)
    )
    
    return Gate2Student(earai_config)