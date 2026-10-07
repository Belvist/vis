"""Minimal Gate 1 Decoder - only ObjectDecoder + CLIP projection"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class Gate1ObjectDecoder(nn.Module):
    """
    Minimal ObjectDecoder for Gate 1.
    Predicts: class logits, bbox coordinates, objectness
    """
    
    def __init__(self, 
                 token_dim: int = 256,
                 num_classes: int = 80,  # COCO classes
                 num_queries: int = 16,  # matches visual tokens
                 hidden_dim: int = 256):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Token-to-query projection
        self.query_proj = nn.Linear(token_dim, hidden_dim)
        
        # Class prediction head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        )
        
        # Bbox regression head (cx, cy, w, h) normalized
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # Objectness score
        self.obj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Token position embeddings (learned)
        self.pos_embed = nn.Parameter(torch.randn(16, hidden_dim) * 0.02)
        
    def forward(self, tokens: torch.Tensor) -> Dict:
        """
        tokens: [B, N, D] visual tokens
        Returns: dict with class_logits, bboxes, objectness, detections
        """
        B, N, D = tokens.shape
        
        # Add position embeddings
        tokens = tokens + self.pos_embed[:N].unsqueeze(0)
        
        # Project to query space
        queries = self.query_proj(tokens)  # [B, N, hidden_dim]
        
        # Predictions
        class_logits = self.class_head(queries)      # [B, N, num_classes+1]
        bboxes_cxcywh = torch.sigmoid(self.bbox_head(queries))  # [B, N, 4] normalized
        objectness = self.obj_head(queries).squeeze(-1)  # [B, N]
        
        # Convert to normalized xyxy for loss compatibility
        cx, cy, w, h = bboxes_cxcywh.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        bboxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
        
        # Detections format for loss (normalized xyxy)
        detections = []
        for b in range(B):
            det = {
                'boxes': bboxes_xyxy[b].detach().cpu().numpy(),  # [N, 4] normalized xyxy
                'scores': objectness[b].detach().cpu().numpy(),
                'labels': class_logits[b].argmax(-1).detach().cpu().numpy()  # [N]
            }
            detections.append(det)
        
        return {
            'class_logits': class_logits,
            'bboxes': bboxes_cxcywh,           # [cx, cy, w, h] normalized
            'bboxes_xyxy': bboxes_xyxy,        # [x1, y1, x2, y2] normalized
            'objectness': objectness,
            'detections': detections,          # For loss compatibility
        }


class Gate1Decoder(nn.Module):
    """
    Minimal Gate 1 Decoder - only ObjectDecoder + CLIP projection
    """
    
    def __init__(self, 
                 token_dim: int = 256,
                 num_classes: int = 80,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.object_decoder = Gate1ObjectDecoder(
            token_dim, num_classes=num_classes, hidden_dim=hidden_dim
        )
        
        # CLIP projection (trainable, part of decoder)
        self.clip_proj = nn.Linear(token_dim, 512)
    
    def forward(self, 
                tokens: torch.Tensor) -> Dict:
        """
        Full decode from visual tokens.
        Returns all predictions.
        """
        # Object detection
        obj_out = self.object_decoder(tokens)
        
        # CLIP projection
        clip_proj = self.clip_proj(tokens.mean(dim=1))  # [B, 512]
        
        return {
            'class_logits': obj_out['class_logits'],
            'bboxes': obj_out['bboxes'],
            'bboxes_xyxy': obj_out['bboxes_xyxy'],
            'objectness': obj_out['objectness'],
            'detections': obj_out['detections'],
            'clip_proj': clip_proj,
        }


def create_gate1_decoder(config: dict) -> Gate1Decoder:
    """Factory for minimal Gate 1 decoder"""
    return Gate1Decoder(
        token_dim=config.get('token_dim', 256),
        num_classes=config.get('num_classes', 80),
        hidden_dim=config.get('hidden_dim', 256)
    )