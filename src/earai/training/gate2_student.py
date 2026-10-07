"""Gate 2 student: backbone + adaptive UI tokens + UI decoder."""
import torch
import torch.nn as nn
from typing import Dict

from earai.core.config import EarAIConfig
from earai.models.backbone import create_backbone
from earai.heads.adaptive_tokens import create_adaptive_pooler
from earai.training.gate2_decoder import Gate2Decoder


class Gate2Student(nn.Module):
    def __init__(self, earai_config: EarAIConfig, train_config: dict):
        super().__init__()
        self.config = earai_config
        self.train_config = train_config
        self.num_tokens = int(train_config.get("num_scene_tokens", 32))
        if self.num_tokens < 4:
            raise ValueError("num_scene_tokens must be >= 4")

        self.backbone = create_backbone(self.config)

        base = self.num_tokens // 4
        rem = self.num_tokens % 4
        tokens_per_scale = [base + (1 if i < rem else 0) for i in range(4)]
        self.token_pooler = create_adaptive_pooler({
            "type": "multiscale_tokenlearner",
            "channels_list": [self.config.feature_dim] * 4,
            "num_tokens_per_scale": tokens_per_scale,
            "bottleneck_dim": 64,
        })

        self.decoder = Gate2Decoder(
            token_dim=self.config.feature_dim,
            num_classes=int(train_config.get("num_ui_classes", 15)),
            num_queries=self.num_tokens,
            hidden_dim=int(train_config.get("hidden_dim", 256)),
            style_dim=int(train_config.get("style_dim", 10)),
        )

    def forward(self, x: torch.Tensor) -> Dict:
        features_dict = self.backbone(x)
        features = [
            features_dict["F4"],
            features_dict["F8"],
            features_dict["F16"],
            features_dict["F32"],
        ]
        token_result = self.token_pooler(features, return_attention=False)
        tokens = token_result["tokens"] if isinstance(token_result, dict) else token_result
        if tokens.shape[1] != self.num_tokens:
            raise RuntimeError(
                f"Gate2 token contract broken: expected {self.num_tokens}, got {tokens.shape[1]}"
            )
        return {"tokens": tokens, **self.decoder(tokens)}


def create_gate2_student(config: dict) -> nn.Module:
    earai_config = EarAIConfig(
        backbone_pretrained=config.get("backbone_pretrained", True),
        feature_dim=int(config.get("feature_dim", 256)),
        num_scene_tokens=int(config.get("num_scene_tokens", 32)),
        peripheral_resolution=tuple(config.get("peripheral_resolution", (224, 224))),
        fovea_resolution=tuple(config.get("fovea_resolution", (600, 600))),
        max_entities=int(config.get("max_entities", 32)),
    )
    return Gate2Student(earai_config, config)
