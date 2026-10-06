"""Lightweight heads for Vera Eye"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..core.config import VeraEyeConfig


class ObjectRegionHead(nn.Module):
    """Detects objects/regions from spatial features"""
    def __init__(self, config: VeraEyeConfig):
        super().__init__()
        self.config = config
        in_dim = 576  # MobileNetV3 feature dim
        hidden = config.object_head_dim

        self.conv1 = nn.Conv2d(in_dim, hidden, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, hidden)

        # Detection heads (per spatial location)
        self.cls_head = nn.Conv2d(hidden, 1, 1)  # objectness
        self.bbox_head = nn.Conv2d(hidden, 4, 1)  # x1, y1, x2, y2 (normalized)
        self.embed_head = nn.Conv2d(hidden, config.entity_embedding_dim, 1)  # visual embedding

    def forward(self, spatial_features: torch.Tensor) -> dict:
        """
        spatial_features: [B, 576, 7, 7]
        Returns: objectness [B, 1, 7, 7], bboxes [B, 4, 7, 7], embeddings [B, D, 7, 7]
        """
        x = F.relu(self.norm1(self.conv1(spatial_features)))
        x = F.relu(self.norm2(self.conv2(x)))

        objectness = torch.sigmoid(self.cls_head(x))
        bboxes = torch.sigmoid(self.bbox_head(x))  # Normalized 0-1
        embeddings = self.embed_head(x)  # [B, D, 7, 7]

        return {
            "objectness": objectness,
            "bboxes": bboxes,
            "embeddings": embeddings
        }


class SemanticHead(nn.Module):
    """Semantic classification + scene-level embedding"""
    def __init__(self, config: VeraEyeConfig, num_classes: int = 80):
        super().__init__()
        self.config = config
        in_dim = config.feature_dim  # Global embedding dim
        hidden = config.semantic_head_dim

        self.fc1 = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.classifier = nn.Linear(hidden, num_classes)
        self.scene_projector = nn.Linear(hidden, config.entity_embedding_dim)

    def forward(self, global_embedding: torch.Tensor) -> dict:
        """
        global_embedding: [B, D]
        Returns: class_logits [B, num_classes], scene_embedding [B, D]
        """
        x = F.relu(self.norm1(self.fc1(global_embedding)))
        x = F.relu(self.norm2(self.fc2(x)))

        class_logits = self.classifier(x)
        scene_embedding = self.scene_projector(x)

        return {
            "class_logits": class_logits,
            "scene_embedding": scene_embedding
        }


class MotionTrackerHead(nn.Module):
    """Predicts motion vectors for tracked entities"""
    def __init__(self, config: VeraEyeConfig):
        super().__init__()
        self.config = config
        # Input: entity visual embedding + previous motion
        in_dim = config.entity_embedding_dim + 2  # embedding + prev motion
        hidden = config.motion_head_dim

        self.fc1 = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.motion_head = nn.Linear(hidden, 2)  # dx, dy

    def forward(self, entity_embeddings: torch.Tensor, prev_motion: torch.Tensor) -> torch.Tensor:
        """
        entity_embeddings: [B, N, D]
        prev_motion: [B, N, 2]
        Returns: motion_vectors [B, N, 2]
        """
        B, N, D = entity_embeddings.shape
        x = torch.cat([entity_embeddings, prev_motion], dim=-1)  # [B, N, D+2]
        x = x.view(B * N, D + 2)

        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        motion = self.motion_head(x)  # [B*N, 2]

        return motion.view(B, N, 2)


class OCRHead(nn.Module):
    """Lightweight text detection + recognition"""
    def __init__(self, config: VeraEyeConfig, vocab_size: int = 66):  # alphanumeric + punct
        super().__init__()
        self.config = config
        in_dim = 576
        hidden = config.ocr_head_dim

        # Text detection (spatial)
        self.det_conv1 = nn.Conv2d(in_dim, hidden, 3, padding=1)
        self.det_norm = nn.GroupNorm(4, hidden)
        self.det_conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.textness = nn.Conv2d(hidden, 1, 1)
        self.text_bbox = nn.Conv2d(hidden, 4, 1)

        # Text recognition (per detected region - simplified)
        self.rec_fc1 = nn.Linear(in_dim, hidden * 2)
        self.rec_norm = nn.LayerNorm(hidden * 2)
        self.rec_fc2 = nn.Linear(hidden * 2, hidden)
        self.rec_out = nn.Linear(hidden, vocab_size * 30)  # 30 chars max

    def forward(self, spatial_features: torch.Tensor, global_embedding: torch.Tensor) -> dict:
        """
        Returns text detection + recognition logits
        """
        B = spatial_features.shape[0]

        # Detection
        x = F.relu(self.det_norm(self.det_conv1(spatial_features)))
        x = F.relu(self.det_conv2(x))
        textness = torch.sigmoid(self.textness(x))
        text_bboxes = torch.sigmoid(self.text_bbox(x))

        # Recognition (simplified - use global for now)
        rec = F.relu(self.rec_norm(self.rec_fc1(global_embedding)))
        rec = F.relu(self.rec_fc2(rec))
        rec_logits = self.rec_out(rec).view(B, 30, -1)  # [B, seq_len, vocab]

        return {
            "textness": textness,
            "text_bboxes": text_bboxes,
            "rec_logits": rec_logits
        }


class ChangeHead(nn.Module):
    """Classifies scene changes"""
    def __init__(self, config: VeraEyeConfig):
        super().__init__()
        self.config = config
        in_dim = config.feature_dim * 2  # current + prev global embedding
        hidden = config.change_head_dim

        self.fc1 = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 5)  # appeared, disappeared, moved, changed, occluded

    def forward(self, curr_embedding: torch.Tensor, prev_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([curr_embedding, prev_embedding], dim=-1)
        x = F.relu(self.norm1(self.fc1(x)))
        return self.fc2(x)


def create_heads(config: VeraEyeConfig) -> dict:
    heads = {
        "object_region": ObjectRegionHead(config),
        "semantic": SemanticHead(config),
        "motion_tracker": MotionTrackerHead(config),
        "ocr": OCRHead(config),
        "change": ChangeHead(config)
    }
    total = sum(sum(p.numel() for p in h.parameters()) for h in heads.values())
    print(f"Heads total params: {total / 1e6:.2f}M")
    return heads