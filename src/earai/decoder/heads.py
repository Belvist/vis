"""Decoder heads for EarAI student model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ObjectDecoder(nn.Module):
    """
    Object detection decoder from visual tokens.
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
        Returns: dict with class_logits, bboxes, objectness, and detections (for loss compatibility)
        """
        B, N, D = tokens.shape
        
        # Add position embeddings
        tokens = tokens + self.pos_embed[:N].unsqueeze(0)
        
        # Project to query space
        queries = self.query_proj(tokens)  # [B, N, hidden_dim]
        
        # Predictions
        class_logits = self.class_head(queries)      # [B, N, num_classes+1]
        bboxes_cxcywh = torch.sigmoid(self.bbox_head(queries))  # [B, N, 4] normalized [cx, cy, w, h]
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


class RelationDecoder(nn.Module):
    """Predict relations between entities (holding, on, near, looking_at, etc.)"""
    
    def __init__(self, 
                 token_dim: int = 256,
                 num_relations: int = 8,  # holding, on, near, left_of, right_of, above, below, inside
                 hidden_dim: int = 256):
        super().__init__()
        self.num_relations = num_relations
        
        # Pairwise relation predictor
        self.rel_head = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
            nn.Sigmoid()  # Multi-label
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D]
        Returns: [B, N, N, num_relations] relation probabilities
        """
        B, N, D = tokens.shape
        
        # Create all pairs
        tokens_i = tokens.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        tokens_j = tokens.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        
        pairs = torch.cat([tokens_i, tokens_j], dim=-1)  # [B, N, N, 2D]
        
        # Predict relations
        rel_logits = self.rel_head(pairs)  # [B, N, N, num_relations]
        
        # Mask diagonal (no self-relations)
        mask = torch.eye(N, device=tokens.device).bool()
        rel_logits = rel_logits.masked_fill(mask.unsqueeze(0).unsqueeze(-1), 0)
        
        return rel_logits


class TextDecoder(nn.Module):
    """OCR decoder - reads text from visual tokens in text regions"""
    
    def __init__(self, 
                 token_dim: int = 256,
                 vocab_size: int = 5000,  # Character vocabulary
                 max_len: int = 32,
                 hidden_dim: int = 256):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Text region encoder
        self.region_encoder = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Autoregressive decoder
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(max_len, hidden_dim) * 0.02)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, 8, hidden_dim*4, batch_first=True),
            num_layers=4
        )
        
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, 
                region_tokens: torch.Tensor,  # [B, N_regions, D]
                target_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        region_tokens: [B, N_regions, D] tokens for text regions
        target_seq: [B, L] for teacher forcing (training)
        Returns: logits [B, N_regions, L, vocab_size]
        """
        B, N_reg, D = region_tokens.shape
        
        # Encode regions
        memory = self.region_encoder(region_tokens)  # [B, N_reg, hidden_dim]
        memory = memory.view(B * N_reg, 1, -1)  # [B*N_reg, 1, hidden_dim]
        
        if target_seq is not None:
            # Training with teacher forcing
            L = target_seq.shape[1]
            tgt = self.embed(target_seq) + self.pos_embed[:L].unsqueeze(0)
            tgt = tgt.view(B * N_reg, L, -1)
            
            out = self.decoder(tgt, memory)  # [B*N_reg, L, hidden]
            logits = self.output_proj(out)  # [B*N_reg, L, vocab]
            return logits.view(B, N_reg, -1, self.vocab_size)
        else:
            # Inference: greedy decoding
            return self._generate(memory)
    
    def _generate(self, memory: torch.Tensor) -> torch.Tensor:
        """Greedy generation for inference"""
        B_N, _, D = memory.shape
        device = memory.device
        
        # Start token
        start_token = torch.full((B_N, 1), 1, dtype=torch.long, device=device)  # SOS=1
        
        generated = []
        for _ in range(self.max_len):
            tgt = self.embed(start_token) if len(generated) == 0 else self.embed(torch.tensor(generated, device=device).unsqueeze(0))
            # Simplified - in practice use proper generation loop
            pass
        
        return torch.zeros(B_N, self.max_len, self.vocab_size, device=device)


class GroundingDecoder(nn.Module):
    """
    Links visual tokens to language (CLIP-style grounding).
    Predicts which tokens correspond to which text phrases.
    """
    
    def __init__(self, 
                 token_dim: int = 256,
                 text_dim: int = 512,  # CLIP text dim
                 hidden_dim: int = 256):
        super().__init__()
        
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.grounding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                visual_tokens: torch.Tensor,  # [B, N, D]
                text_embeddings: torch.Tensor) -> torch.Tensor:  # [B, M, text_dim]
        """
        Returns grounding scores [B, N, M] for each token-phrase pair
        """
        B, N, D = visual_tokens.shape
        _, M, T = text_embeddings.shape
        
        v_proj = self.token_proj(visual_tokens)  # [B, N, H]
        t_proj = self.text_proj(text_embeddings)  # [B, M, H]
        
        # Cross attention for grounding
        v_exp = v_proj.unsqueeze(2).expand(-1, -1, M, -1)  # [B, N, M, H]
        t_exp = t_proj.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, H]
        
        combined = torch.cat([v_exp, t_exp], dim=-1)  # [B, N, M, 2H]
        scores = self.grounding_head(combined).squeeze(-1)  # [B, N, M]
        
        return scores


# Full decoder ensemble
class EarAIDecoder(nn.Module):
    """
    Complete decoder for EarAI visual tokens.
    Produces: objects, bboxes, relations, text, grounding
    """
    
    def __init__(self, 
                 token_dim: int = 256,
                 num_classes: int = 80,
                 num_relations: int = 8,
                 vocab_size: int = 5000,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.object_decoder = ObjectDecoder(token_dim, num_classes=num_classes, hidden_dim=hidden_dim)
        self.relation_decoder = RelationDecoder(token_dim, num_relations=num_relations, hidden_dim=hidden_dim)
        self.text_decoder = TextDecoder(token_dim, vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.grounding_decoder = GroundingDecoder(token_dim, hidden_dim=hidden_dim)
    
    def forward(self, 
                tokens: torch.Tensor,           # [B, N, D]
                clip_text_emb: Optional[torch.Tensor] = None,
                text_regions_mask: Optional[torch.Tensor] = None,
                target_text_seq: Optional[torch.Tensor] = None) -> Dict:
        """
        Full decode from visual tokens.
        Returns all predictions.
        """
        # Object detection
        obj_out = self.object_decoder(tokens)
        
        # Relations
        relations = self.relation_decoder(tokens)
        
        # Text (only on text regions if mask provided)
        text_logits = None
        # Would need to identify text regions first
        
        # Grounding (if text embeddings provided)
        grounding = None
        if hasattr(self, 'grounding_decoder') and clip_text_emb is not None:
            grounding = self.grounding_decoder(tokens, clip_text_emb)
        
        return {
            'class_logits': obj_out['class_logits'],
            'bboxes': obj_out['bboxes'],
            'bboxes_xyxy': obj_out['bboxes_xyxy'],
            'objectness': obj_out['objectness'],
            'detections': obj_out['detections'],
            'relations': relations,
            'grounding': grounding,
        }


def create_decoder(config: dict) -> EarAIDecoder:
    """Factory for decoder"""
    return EarAIDecoder(
        token_dim=config.get('token_dim', 256),
        num_classes=config.get('num_classes', 80),
        num_relations=config.get('num_relations', 8),
        vocab_size=config.get('vocab_size', 5000),
        hidden_dim=config.get('hidden_dim', 256)
    )