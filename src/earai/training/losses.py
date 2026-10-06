"""Training losses for EarAI distillation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for EarAI.
    Matches student visual tokens to teacher representations.
    """
    
    def __init__(self, 
                 token_dim: int = 256,
                 num_classes: int = 80,
                 num_relations: int = 8,
                 vocab_size: int = 5000,
                 lambda_obj: float = 1.0,
                 lambda_bbox: float = 2.0,
                 lambda_class: float = 1.0,
                 lambda_rel: float = 0.5,
                 lambda_text: float = 1.0,
                 lambda_grounding: float = 1.0,
                 lambda_clip: float = 1.0,
                 lambda_seg: float = 0.5):
        super().__init__()
        
        self.lambda_obj = lambda_obj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        self.lambda_rel = lambda_rel
        self.lambda_text = lambda_text
        self.lambda_grounding = lambda_grounding
        self.lambda_clip = lambda_clip
        self.lambda_seg = lambda_seg
        
        # Projection for CLIP alignment
        self.clip_proj = nn.Linear(256, 512)
        
        # Segmentation head for distillation
        self.seg_head = nn.Conv2d(256, 21, 1)  # 21 Pascal VOC classes
    
    def forward(self, 
                student_tokens: torch.Tensor,      # [B, N, D]
                teacher_out: Dict,
                student_out: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all distillation losses.
        """
        losses = {}
        
        # 1. CLIP embedding alignment
        if 'clip_embeddings' in teacher_out:
            student_clip = self.clip_proj(student_tokens.mean(dim=1))  # [B, 512]
            teacher_clip = teacher_out['clip_embeddings']  # [B, 512]
            losses['clip'] = F.mse_loss(student_clip, teacher_clip)
        
        # 2. Object detection losses
        if 'detections' in teacher_out and 'detections' in student_out:
            det_losses = self._detection_loss(teacher_out['detections'], student_out)
            losses.update(det_losses)
        
        # 3. Relation losses
        if 'relations' in teacher_out and 'relations' in student_out:
            losses['relation'] = self._relation_loss(teacher_out['relations'], student_out['relations'])
        
        # 3. Segmentation loss
        if 'segmentation' in teacher_out:
            losses['segmentation'] = self._segmentation_loss(teacher_out['segmentation'], student_tokens)
        
        # 4. Grounding loss
        if 'grounding' in teacher_out and 'grounding' in student_out:
            losses['grounding'] = F.binary_cross_entropy(
                student_out['grounding'], teacher_out['grounding']
            )
        
        # Total weighted loss
        total = 0
        for k, v in losses.items():
            weight = getattr(self, f'lambda_{k}', 1.0)
            total += weight * v
        
        losses['total'] = total
        return losses
    
    def _detection_loss(self, teacher_dets: List[Dict], student_out: Dict) -> Dict:
        """Detection distillation loss"""
        losses = {}
        
        # For each image in batch
        for b, (t_det, s_out) in enumerate(zip(teacher_dets, student_out)):
            if not t_det.get('boxes', []):
                continue
            
            t_boxes = torch.tensor(t_det['boxes'], device=student_out['bboxes'].device)
            t_labels = torch.tensor(t_det['labels'], device=student_out['class_logits'].device)
            
            # Match student predictions to teacher boxes (simplified: first N predictions)
            N_pred = student_out['bboxes'].shape[1]
            N_gt = len(t_boxes)
            
            if N_gt == 0:
                continue
            
            # Bbox regression loss (L1 on matched predictions)
            matched = min(N_pred, N_gt)
            if matched > 0:
                losses['bbox'] = F.l1_loss(
                    student_out['bboxes'][b, :matched], t_boxes[:matched]
                )
            
            # Class loss (cross entropy on matched)
            s_logits = student_out['class_logits'][b, :matched]
            losses['class'] = F.cross_entropy(s_logits, t_labels[:matched])
            
            # Objectness loss
            t_obj = torch.ones_like(student_out['objectness'][b, :matched])
            losses['objectness'] = F.binary_cross_entropy(
                student_out['objectness'][b, :matched], t_obj
            )
        
        return losses
    
    def _relation_loss(self, teacher_rel: torch.Tensor, student_rel: torch.Tensor) -> torch.Tensor:
        """Relation distillation loss (binary cross entropy)"""
        return F.binary_cross_entropy(student_rel, teacher_rel)
    
    def _segmentation_loss(self, teacher_seg: torch.Tensor, student_tokens: torch.Tensor) -> torch.Tensor:
        """Segmentation distillation loss"""
        B, N, D = student_tokens.shape
        H = W = int(N ** 0.5)
        
        # Reshape tokens to spatial
        tokens_spatial = student_tokens.transpose(1, 2).view(-1, 256, H, W)
        
        # Project to segmentation logits
        seg_logits = self.seg_head(tokens_spatial)
        
        # Cross entropy with teacher segmentation
        return F.cross_entropy(seg_logits, teacher_seg.long())


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for visual token alignment with CLIP.
    Pulls matching image-text pairs together, pushes others apart.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        image_emb: [B, D] or [B, N, D]
        text_emb: [B, D] or [B, M, D]
        """
        # Normalize
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Similarity matrix
        if image_emb.dim() == 3:  # [B, N, D]
            # Use mean token for contrastive
            image_emb = image_emb.mean(dim=1)
        
        logits = image_emb @ text_emb.t() / self.temperature  # [B, B]
        
        # Labels: diagonal is positive
        labels = torch.arange(len(image_emb), device=image_emb.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2


class StateTransitionLoss(nn.Module):
    """
    State-transition distillation loss.
    Trains student to predict next state from current state + residual.
    """
    
    def __init__(self, 
                 state_dim: int = 256,
                 lambda_state: float = 1.0,
                 lambda_uncertainty: float = 0.5):
        super().__init__()
        self.lambda_state = lambda_state
        self.lambda_uncertainty = lambda_uncertainty
        
        # State transition predictor (teacher)
        self.transition_predictor = nn.Sequential(
            nn.Linear(state_dim * 2, 512),
            nn.GELU(),
            nn.Linear(512, state_dim)
        )
    
    def forward(self, 
                student_state: torch.Tensor,      # [B, N, D]
                teacher_next_state: torch.Tensor,  # [B, N, D]
                residual_features: torch.Tensor) -> Dict:  # [B, M, D]
        """
        Predict next state from current state + residual.
        """
        B, N, D = student_state.shape
        
        # Student predicts next state
        combined = torch.cat([student_state, residual_features.mean(dim=1).unsqueeze(1).expand(-1, N, -1)], dim=-1)
        predicted_next = self.transition_predictor(combined)
        
        # State loss
        state_loss = F.mse_loss(predicted_next, teacher_next_state)
        
        return {
            'state_transition': state_loss,
            'total': state_loss
        }


def create_losses(config: dict) -> Dict[str, nn.Module]:
    """Factory for loss functions"""
    return {
        'distillation': DistillationLoss(**config.get('distillation', {})),
        'contrastive': ContrastiveLoss(**config.get('contrastive', {})),
        'state_transition': StateTransitionLoss(**config.get('state_transition', {})),
    }