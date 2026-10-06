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
        
        # Segmentation loss - DISABLED for Gate 1 (size mismatch with 16 tokens)
        # if 'segmentation' in teacher_out:
        #     losses['segmentation'] = self._segmentation_loss(teacher_out['segmentation'], student_tokens)
        
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
        
        # COCO class mapping: teacher label -> student class index (0-79)
        # FasterRCNN uses COCO category IDs (1-90, sparse)
        # We map to 0-79 continuous indices
        coco_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
            11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
            21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
            33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
            42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
            52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
            61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
            74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
            84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
        }
        
        # For each image in batch
        for b, t_det in enumerate(teacher_dets):
            boxes = t_det.get('boxes', None)
            if boxes is None or (hasattr(boxes, '__len__') and len(boxes) == 0):
                continue
            
            # Teacher boxes in pixel coordinates [x1, y1, x2, y2]
            t_boxes = torch.tensor(boxes, dtype=torch.float32, device=student_out['bboxes_xyxy'].device)
            t_labels_raw = torch.tensor(t_det['labels'], dtype=torch.long, device=student_out['class_logits'].device)
            
            # Map teacher labels to student class indices
            t_labels = torch.tensor([coco_mapping.get(int(l), 0) for l in t_labels_raw], 
                                   dtype=torch.long, device=student_out['class_logits'].device)
            
            # Normalize teacher boxes to 0-1 (assuming teacher boxes are in pixel coordinates)
            # We need image size - for now assume 224x224
            t_boxes_normalized = t_boxes.clone()
            t_boxes_normalized[:, [0, 2]] /= 224.0  # x1, x2
            t_boxes_normalized[:, [1, 3]] /= 224.0  # y1, y2
            t_boxes_normalized = t_boxes_normalized.clamp(0, 1)
            
            # Student predictions (already normalized xyxy)
            s_boxes = student_out['bboxes_xyxy'][b]  # [N, 4] normalized xyxy
            s_logits = student_out['class_logits'][b]  # [N, num_classes+1]
            s_obj = student_out['objectness'][b]  # [N]
            
            N_pred = s_boxes.shape[0]
            N_gt = len(t_boxes_normalized)
            
            if N_gt == 0:
                continue
            
            # Match student predictions to teacher boxes (simplified: first N predictions)
            matched = min(N_pred, N_gt)
            if matched > 0:
                # Bbox regression loss (L1 on matched predictions, normalized xyxy)
                losses['bbox'] = F.l1_loss(s_boxes[:matched], t_boxes_normalized[:matched])
            
            # Class loss (cross entropy on matched)
            s_logits_matched = s_logits[:matched]
            losses['class'] = F.cross_entropy(s_logits_matched, t_labels[:matched])
            
            # Objectness loss
            t_obj = torch.ones_like(s_obj[:matched])
            losses['objectness'] = F.binary_cross_entropy(s_obj[:matched], t_obj)
        
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