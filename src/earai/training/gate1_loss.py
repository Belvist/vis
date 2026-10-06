"""Gate 1 Loss with Hungarian Matching"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    boxes1: [N, 4] xyxy normalized
    boxes2: [M, 4] xyxy normalized
    Returns: [N, M] IoU
    """
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    union = area1[:, None] + area2[None, :] - inter  # [N, M]
    
    iou = inter / (union + 1e-6)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/
    boxes1: [N, 4] xyxy normalized
    boxes2: [M, 4] xyxy normalized
    Returns: [N, M] GIoU
    """
    # IoU
    iou = box_iou(boxes1, boxes2)
    
    # Enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    union = area1[:, None] + area2[None, :] - (iou * (area1[:, None] + area2[None, :] - 1e-6))
    
    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for bipartite matching between predictions and targets.
    Cost: class_cost + 5.0 * bbox_L1 + 2.0 * (1 - GIoU)
    """
    
    def __init__(self, 
                 cost_class: float = 1.0,
                 cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, 
                pred_logits: torch.Tensor,    # [B, N, num_classes+1]
                pred_boxes: torch.Tensor,     # [B, N, 4] normalized xyxy
                targets: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns list of (pred_indices, target_indices) for each batch item
        """
        B, N, _ = pred_logits.shape
        
        # Flatten batch
        pred_logits = pred_logits.flatten(0, 1)  # [B*N, num_classes+1]
        pred_boxes = pred_boxes.flatten(0, 1)    # [B*N, 4]
        
        # Concat targets
        target_boxes = torch.cat([torch.tensor(t['boxes'], dtype=torch.float32) for t in targets])
        target_labels = torch.cat([torch.tensor(t['labels'], dtype=torch.long) for t in targets])
        
        # Split points for each batch
        target_sizes = [len(t['boxes']) for t in targets]
        
        indices = []
        start = 0
        
        for b, size in enumerate(target_sizes):
            if size == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            
            # Get predictions for this batch
            pred_logits_b = pred_logits[b * N:(b + 1) * N]  # [N, C+1]
            pred_boxes_b = pred_boxes[b * N:(b + 1) * N]    # [N, 4]
            
            # Target for this batch
            target_boxes_b = target_boxes[start:start + size].to(pred_boxes.device)  # [M, 4]
            target_labels_b = target_labels[start:start + size].to(pred_logits.device)  # [M]
            start += size
            
            # Classification cost (negative log prob of target class)
            prob = pred_logits_b.softmax(-1)  # [N, C+1]
            cost_class = -prob[:, target_labels_b]  # [N, M]
            
            # Bbox L1 cost
            cost_bbox = torch.cdist(pred_boxes_b, target_boxes_b, p=1)  # [N, M]
            
            # GIoU cost
            giou = generalized_box_iou(pred_boxes_b, target_boxes_b)  # [N, M]
            cost_giou = 1 - giou
            
            # Total cost
            C = (self.cost_class * cost_class + 
                 self.cost_bbox * cost_bbox + 
                 self.cost_giou * cost_giou)  # [N, M]
            
            # Hungarian matching on CPU
            C_np = C.detach().cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(C_np)
            
            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=pred_boxes.device),
                torch.tensor(target_idx, dtype=torch.long, device=pred_boxes.device)
            ))
        
        return indices


class Gate1Loss(nn.Module):
    """
    Gate 1 Loss:
    L = L_class + 5 * L_bbox + 2 * L_giou + L_objectness + L_clip
    """
    
    def __init__(self,
                 num_classes: int = 80,
                 lambda_class: float = 1.0,
                 lambda_bbox: float = 5.0,
                 lambda_giou: float = 2.0,
                 lambda_objectness: float = 1.0,
                 lambda_clip: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_class = lambda_class
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.lambda_objectness = lambda_objectness
        self.lambda_clip = lambda_clip
        
        self.matcher = HungarianMatcher(
            cost_class=lambda_class,
            cost_bbox=lambda_bbox,
            cost_giou=lambda_giou
        )
        
        # CLIP projection (trainable, part of model)
        self.clip_proj = nn.Linear(256, 512)
    
    def forward(self,
                student_tokens: torch.Tensor,      # [B, N, 256]
                student_out: Dict,                  # decoder output
                teacher_out: Dict) -> Dict:
        """
        Compute Gate 1 loss.
        """
        losses = {}
        
        # Get predictions
        pred_logits = student_out['class_logits']      # [B, N, C+1]
        pred_boxes = student_out['bboxes_xyxy']        # [B, N, 4] normalized xyxy
        pred_obj = student_out['objectness']           # [B, N]
        
        # Teacher detections
        teacher_dets = teacher_out['detections']       # List[Dict]
        teacher_clip = teacher_out['clip_embeddings']  # [B, 512]
        
        B, N, _ = pred_logits.shape
        
        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, teacher_dets)
        
        # Accumulate losses across batch
        loss_class = 0
        loss_bbox = 0
        loss_giou = 0
        loss_objectness = 0
        num_matched = 0
        
        for b, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                # No ground truth objects - all predictions should be background
                target_labels_bg = torch.full((N,), self.num_classes, 
                                             dtype=torch.long, device=pred_logits.device)
                loss_class += F.cross_entropy(pred_logits[b], target_labels_bg)
                loss_objectness += F.binary_cross_entropy(pred_obj[b], torch.zeros_like(pred_obj[b]))
                continue
            
            # Matched predictions
            pred_logits_matched = pred_logits[b, pred_idx]      # [M, C+1]
            pred_boxes_matched = pred_boxes[b, pred_idx]        # [M, 4]
            pred_obj_matched = pred_obj[b, pred_idx]            # [M]
            
            # Target boxes and labels
            target_boxes = torch.tensor(
                teacher_dets[b]['boxes'], dtype=torch.float32, device=pred_boxes.device
            )[target_idx]  # [M, 4]
            target_labels = torch.tensor(
                teacher_dets[b]['labels'], dtype=torch.long, device=pred_logits.device
            )[target_idx]  # [M]
            
            # Class loss (cross entropy)
            loss_class += F.cross_entropy(pred_logits_matched, target_labels)
            
            # Bbox L1 loss
            loss_bbox += F.l1_loss(pred_boxes_matched, target_boxes)
            
            # GIoU loss
            giou = generalized_box_iou(pred_boxes_matched, target_boxes)
            loss_giou += (1 - giou).mean()
            
            # Objectness loss (matched = 1)
            loss_objectness += F.binary_cross_entropy(
                pred_obj_matched, torch.ones_like(pred_obj_matched)
            )
            
            num_matched += len(pred_idx)
            
            # Unmatched predictions -> background
            all_pred_idx = torch.arange(N, device=pred_logits.device)
            unmatched = all_pred_idx[~torch.isin(all_pred_idx, pred_idx)]
            
            if len(unmatched) > 0:
                loss_class += F.cross_entropy(
                    pred_logits[b, unmatched],
                    torch.full((len(unmatched),), self.num_classes, dtype=torch.long, device=pred_logits.device)
                )
                loss_objectness += F.binary_cross_entropy(
                    pred_obj[b, unmatched],
                    torch.zeros(len(unmatched), device=pred_obj.device)
                )
        
        # Normalize by number of matched objects (or batch size)
        norm = max(num_matched, B)
        loss_class = loss_class / B
        loss_bbox = loss_bbox / max(num_matched, 1)
        loss_giou = loss_giou / max(num_matched, 1)
        loss_objectness = loss_objectness / B
        
        # CLIP alignment loss
        student_clip = self.clip_proj(student_tokens.mean(dim=1))  # [B, 512]
        loss_clip = F.mse_loss(student_clip, teacher_clip)
        
        # Total loss
        total = (self.lambda_class * loss_class +
                 self.lambda_bbox * loss_bbox +
                 self.lambda_giou * loss_giou +
                 self.lambda_objectness * loss_objectness +
                 self.lambda_clip * loss_clip)
        
        return {
            'class': loss_class,
            'bbox': loss_bbox,
            'giou': loss_giou,
            'objectness': loss_objectness,
            'clip': loss_clip,
            'total': total,
            'num_matched': num_matched
        }


def create_gate1_loss(config: dict) -> Gate1Loss:
    """Factory for Gate 1 loss"""
    return Gate1Loss(
        num_classes=config.get('num_classes', 80),
        lambda_class=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        lambda_objectness=1.0,
        lambda_clip=1.0
    )