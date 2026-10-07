"""Gate 2 Decoder - UI-specific decoder with style and hierarchy"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class Gate2ObjectDecoder(nn.Module):
    """
    UI Object decoder from visual tokens.
    Predicts: class logits, bbox coordinates (cx,cy,w,h), objectness, style, hierarchy
    """

    def __init__(self,
                 token_dim: int = 256,
                 num_classes: int = 15,  # UI classes
                 num_queries: int = 32,
                 hidden_dim: int = 256,
                 style_dim: int = 10):  # bg(3), fg(3), radius(1), font_size(1), font_weight(1), line_height(1)
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.style_dim = style_dim

        # Token-to-query projection
        self.query_proj = nn.Linear(token_dim, hidden_dim)

        # Class prediction head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        )

        # Bbox regression head (cx, cy, w, h) -> sigmoid -> valid normalized xyxy
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

        # Style prediction head (all 10 params in [0,1] after sigmoid)
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, style_dim)
        )

        # Hierarchy prediction (parent attention)
        self.hierarchy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # CLIP projection
        self.clip_proj = nn.Linear(token_dim, 512)

        # Token position embeddings
        self.pos_embed = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

    def _cxcywh_to_xyxy(self, boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        """Convert cx,cy,w,h to x1,y1,x2,y2"""
        cx, cy, w, h = boxes_cxcywh.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, tokens: torch.Tensor) -> Dict:
        """
        tokens: [B, N, D] visual tokens
        Returns: dict with class_logits, bboxes_xyxy, objectness, style, hierarchy, clip_proj
        """
        B, N, D = tokens.shape

        # Add position embeddings
        tokens = tokens + self.pos_embed[:N].unsqueeze(0)

        # Project to query space
        queries = self.query_proj(tokens)  # [B, N, hidden_dim]

        # Predictions
        class_logits = self.class_head(queries)      # [B, N, num_classes+1]
        bbox_cxcywh = torch.sigmoid(self.bbox_head(queries))  # [B, N, 4] normalized
        bboxes_xyxy = self._cxcywh_to_xyxy(bbox_cxcywh)
        # Clamp to valid range
        bboxes_xyxy = bboxes_xyxy.clamp(0, 1)

        objectness = self.obj_head(queries).squeeze(-1)  # [B, N]
        style = torch.sigmoid(self.style_head(queries))  # [B, N, style_dim] in [0,1]

        # Hierarchy prediction (pairwise)
        queries_i = queries.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        queries_j = queries.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        pairs = torch.cat([queries_i, queries_j], dim=-1)  # [B, N, N, 2D]
        hierarchy = self.hierarchy_head(pairs).squeeze(-1)  # [B, N, N]

        # Mask diagonal (no self-parent)
        mask = torch.eye(N, device=tokens.device).bool()
        hierarchy = hierarchy.masked_fill(mask.unsqueeze(0), 0)

        # CLIP projection (mean pooled tokens)
        clip_proj = self.clip_proj(tokens.mean(dim=1))

        return {
            'class_logits': class_logits,
            'bboxes_xyxy': bboxes_xyxy,
            'objectness': objectness,
            'style': style,
            'hierarchy': hierarchy,
            'clip_proj': clip_proj,
        }


class Gate2Decoder(nn.Module):
    """
    Complete Gate 2 decoder for UI understanding.
    """

    def __init__(self,
                 token_dim: int = 256,
                 num_classes: int = 15,
                 num_queries: int = 32,
                 hidden_dim: int = 256,
                 style_dim: int = 10):
        super().__init__()

        self.object_decoder = Gate2ObjectDecoder(
            token_dim, num_classes=num_classes, num_queries=num_queries,
            hidden_dim=hidden_dim, style_dim=style_dim
        )

    def forward(self,
                tokens: torch.Tensor) -> Dict:
        """
        Full decode from visual tokens.
        Returns all predictions including clip_proj.
        """
        return self.object_decoder(tokens)


def create_gate2_decoder(config: dict) -> Gate2Decoder:
    """Factory for Gate 2 decoder"""
    return Gate2Decoder(
        token_dim=config.get('token_dim', 256),
        num_classes=config.get('num_ui_classes', 15),
        num_queries=config.get('num_scene_tokens', 32),  # Increased from 16
        hidden_dim=config.get('hidden_dim', 256),
        style_dim=config.get('style_dim', 10)  # 10 params
    )