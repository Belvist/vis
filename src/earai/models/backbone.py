"""Tiny visual backbone - MobileNetV3-Small ~2.5M params"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from ..core.config import EarAIConfig


class TinyBackbone(nn.Module):
    """
    MobileNetV3-Small based backbone
    Output: 576-dim feature vector + 7x7 feature map for heads + ImageNet logits
    """

    def __init__(self, config: EarAIConfig):
        super().__init__()
        self.config = config

        # Load MobileNetV3-Small with pretrained weights
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config.backbone_pretrained else None)

        # Keep original classifier for ImageNet classification
        self.imagenet_classifier = backbone.classifier

        # Remove classifier from features
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature projection to target dim
        self.feature_dim = config.feature_dim
        self.projector = nn.Sequential(
            nn.Linear(576, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.ReLU(inplace=True)
        )

        # For foveated vision - keep spatial features
        self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B, 3, H, W] normalized ImageNet stats
        Returns: dict with global_embedding [B, D], spatial_features [B, D, 7, 7], imagenet_logits [B, 1000]
        """
        # Backbone features
        feat = self.features(x)  # [B, 576, H/32, W/32]

        # Global embedding for EarAI heads
        global_feat = self.avgpool(feat).flatten(1)  # [B, 576]
        global_embedding = self.projector(global_feat)  # [B, D]

        # ImageNet classification (original classifier)
        imagenet_logits = self.imagenet_classifier(global_feat)  # [B, 1000]

        # Spatial features for region heads
        spatial = self.spatial_pool(feat)  # [B, 576, 7, 7]

        return {
            "global_embedding": global_embedding,
            "spatial_features": spatial,
            "raw_features": feat,
            "imagenet_logits": imagenet_logits
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def quantize_int8(self):
        """Post-training dynamic quantization"""
        self.features = torch.quantization.quantize_dynamic(
            self.features, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
        )
        self.projector = torch.quantization.quantize_dynamic(
            self.projector, {nn.Linear}, dtype=torch.qint8
        )
        self.imagenet_classifier = torch.quantization.quantize_dynamic(
            self.imagenet_classifier, {nn.Linear}, dtype=torch.qint8
        )
        return self


def create_backbone(config: EarAIConfig) -> TinyBackbone:
    model = TinyBackbone(config)
    print(f"Backbone params: {model.get_param_count() / 1e6:.2f}M")
    return model