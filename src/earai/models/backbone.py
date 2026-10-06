"""Multi-scale backbone - MobileNetV4-style trunk with F4/F8/F16/F32 features"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional
from ..core.config import EarAIConfig


class MultiScaleBackbone(nn.Module):
    """
    MobileNetV3-Small based backbone with multi-scale feature outputs.
    Returns features at multiple strides for adaptive token pooling.
    """
    
    def __init__(self, config: EarAIConfig):
        super().__init__()
        self.config = config
        
        # Load MobileNetV3-Small as base (V4 not in torchvision yet)
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config.backbone_pretrained else None
        )
        
        self.features = backbone.features
        
        # Stride indices for MobileNetV3-Small (where resolution changes)
        self.stride_indices = {
            4: 1,   # after layer 1: 16 channels, stride 4
            8: 2,   # after layer 2: 24 channels, stride 8
            16: 4,  # after layer 4: 40 channels, stride 16
            32: 9   # after layer 9: 96 channels, stride 32
        }
        
        # Feature dims at each stage
        self.stage_channels = {
            4: 16,
            8: 24,
            16: 40,
            32: 96
        }
        
        # Projectors to common dimension
        self.projectors = nn.ModuleDict({
            'F4': nn.Conv2d(16, config.feature_dim, 1),
            'F8': nn.Conv2d(24, config.feature_dim, 1),
            'F16': nn.Conv2d(40, config.feature_dim, 1),
            'F32': nn.Conv2d(96, config.feature_dim, 1)
        })
        
        # Global pooling for scene embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.scene_projector = nn.Sequential(
            nn.Linear(576, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU()
        )
        
        # Keep ImageNet classifier
        self.imagenet_classifier = backbone.classifier
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 3, H, W] normalized
        Returns dict with multi-scale features
        """
        features = {}
        
        # Run through backbone, capturing intermediate features
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture at stride points
            if i == self.stride_indices[4]:
                features['F4'] = self.projectors['F4'](x)
            elif i == self.stride_indices[8]:
                features['F8'] = self.projectors['F8'](x)
            elif i == self.stride_indices[16]:
                features['F16'] = self.projectors['F16'](x)
            elif i == self.stride_indices[32]:
                features['F32'] = self.projectors['F32'](x)
        
        # Global features
        global_feat = self.global_pool(x).flatten(1)  # [B, 576]
        features['global'] = self.scene_projector(global_feat)  # [B, D]
        
        # ImageNet logits
        features['imagenet_logits'] = self.imagenet_classifier(global_feat)
        
        return features
    
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def quantize_int8(self):
        self.features = torch.quantization.quantize_dynamic(
            self.features, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
        )
        for proj in self.projectors.values():
            proj = torch.quantization.quantize_dynamic(proj, {nn.Conv2d}, dtype=torch.qint8)
        self.scene_projector = torch.quantization.quantize_dynamic(
            self.scene_projector, {nn.Linear}, dtype=torch.qint8
        )
        self.imagenet_classifier = torch.quantization.quantize_dynamic(
            self.imagenet_classifier, {nn.Linear}, dtype=torch.qint8
        )
        return self


def create_backbone(config: EarAIConfig) -> MultiScaleBackbone:
    model = MultiScaleBackbone(config)
    print(f"Backbone params: {model.get_param_count() / 1e6:.2f}M")
    return model