"""Gate 2 Loss - UI-specific distillation losses"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes [N, 4] and [M, 4] normalized xyxy"""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU from https://giou.stanford.edu/"""
    iou = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter

    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """Hungarian matcher for UI element matching"""

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
                pred_logits: torch.Tensor,
                pred_boxes: torch.Tensor,
                targets: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns list of (pred_indices, target_indices) for each batch item
        """
        B, N, _ = pred_logits.shape

        indices = []
        for b in range(B):
            target_elements = targets[b]['ui_elements']
            if not isinstance(target_elements, list):
                target_elements = target_elements.tolist() if hasattr(target_elements, 'tolist') else list(target_elements)

            if len(target_elements) == 0:
                indices.append((torch.empty(0, dtype=torch.long, device=pred_logits.device),
                               torch.empty(0, dtype=torch.long, device=pred_logits.device)))
                continue

            target_boxes = torch.tensor(
                [e['bbox'].tolist() if hasattr(e['bbox'], 'tolist') else e['bbox'] for e in target_elements],
                dtype=torch.float32, device=pred_boxes.device
            )
            target_labels = torch.tensor(
                [e['class_id'] for e in target_elements],
                dtype=torch.long, device=pred_logits.device
            )

            pred_logits_b = pred_logits[b]
            prob = pred_logits_b.softmax(-1)
            cost_class = -prob[:, target_labels]

            cost_bbox = torch.cdist(pred_boxes[b], target_boxes, p=1)

            giou = generalized_box_iou(pred_boxes[b], target_boxes)
            cost_giou = 1 - giou

            C = (self.cost_class * cost_class +
                 self.cost_bbox * cost_bbox +
                 self.cost_giou * cost_giou)

            C_np = C.detach().cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(C_np)

            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=pred_boxes.device),
                torch.tensor(target_idx, dtype=torch.long, device=pred_boxes.device)
            ))

        return indices


class Gate2Loss(nn.Module):
    """
    Gate 2 Loss - UI-specific distillation losses:
    L = L_type + 5×L_bbox + 2×L_giou + L_objectness + L_style + L_hierarchy + L_clip
    """

    def __init__(self,
                 num_classes: int = 15,
                 lambda_type: float = 1.0,
                 lambda_bbox: float = 5.0,
                 lambda_giou: float = 2.0,
                 lambda_objectness: float = 1.0,
                 lambda_style: float = 1.0,
                 lambda_hierarchy: float = 1.0,
                 lambda_clip: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_type = lambda_type
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.lambda_objectness = lambda_objectness
        self.lambda_style = lambda_style
        self.lambda_hierarchy = lambda_hierarchy
        self.lambda_clip = lambda_clip

        self.matcher = HungarianMatcher(
            cost_class=lambda_type,
            cost_bbox=lambda_bbox,
            cost_giou=lambda_giou
        )

        self.clip_proj = nn.Linear(256, 512)

    def forward(self,
                student_tokens: torch.Tensor,
                student_out: Dict,
                teacher_out: Dict) -> Dict:
        """
        Compute Gate 2 loss.
        """
        pred_logits = student_out['class_logits']
        pred_boxes = student_out['bboxes_xyxy']
        pred_obj = student_out['objectness']
        pred_style = student_out.get('style', None)
        pred_hierarchy = student_out.get('hierarchy', None)

        teacher_targets = teacher_out['targets']
        teacher_clip = teacher_out['clip_embeddings']

        B, N, _ = pred_logits.shape

        indices = self.matcher(pred_logits, pred_boxes, teacher_targets)

        loss_type = 0
        loss_bbox = 0
        loss_giou = 0
        loss_objectness = 0
        loss_style = 0
        loss_hierarchy = torch.tensor(0.0, device=pred_logits.device)
        num_matched = 0

        for b, (pred_idx, target_idx) in enumerate(indices):
            target_elements = teacher_targets[b]['ui_elements']
            if not isinstance(target_elements, list):
                target_elements = target_elements.tolist() if hasattr(target_elements, 'tolist') else list(target_elements)

            if len(target_elements) == 0:
                target_labels_bg = torch.full((N,), self.num_classes,
                                             dtype=torch.long, device=pred_logits.device)
                loss_type += F.cross_entropy(pred_logits[b], target_labels_bg)
                loss_objectness += F.binary_cross_entropy(pred_obj[b], torch.zeros_like(pred_obj[b]))
                continue

            pred_logits_matched = pred_logits[b, pred_idx]
            pred_boxes_matched = pred_boxes[b, pred_idx]
            pred_obj_matched = pred_obj[b, pred_idx]
            pred_style_matched = pred_style[b, pred_idx] if pred_style is not None else None
            pred_hierarchy_matched = pred_hierarchy[b, pred_idx, :] if pred_hierarchy is not None else None

            target_elements = teacher_targets[b]['ui_elements']
            if not isinstance(target_elements, list):
                target_elements = target_elements.tolist() if hasattr(target_elements, 'tolist') else list(target_elements)

            target_boxes = torch.tensor(
                [target_elements[i]['bbox'].tolist() if hasattr(target_elements[i]['bbox'], 'tolist') else target_elements[i]['bbox'] for i in target_idx],
                dtype=torch.float32, device=pred_boxes.device
            )
            target_labels = torch.tensor(
                [target_elements[i]['class_id'] for i in target_idx],
                dtype=torch.long, device=pred_logits.device
            )
            target_styles = [target_elements[i].get('style', {}) for i in target_idx]
            target_parent_ids = [target_elements[i].get('parent_id') for i in target_idx]

            # Class loss
            loss_type += F.cross_entropy(pred_logits[b, pred_idx], target_labels)

            # Bbox L1 loss (sum, normalize at end)
            loss_bbox += F.l1_loss(pred_boxes[b, pred_idx], target_boxes, reduction='sum')

            # GIoU loss - only on matched pairs (diagonal)
            giou_matrix = generalized_box_iou(pred_boxes[b, pred_idx], target_boxes)
            giou_diag = torch.diag(giou_matrix)
            loss_giou += (1 - giou_diag).sum()

            # Objectness loss (matched = 1)
            loss_objectness += F.binary_cross_entropy(
                pred_obj[b, pred_idx], torch.ones_like(pred_obj[b, pred_idx])
            )

            num_matched += len(pred_idx)

            # Style loss - full 10 dims
            if pred_style is not None:
                # target_styles from dataset are already tensors [M, 10]
                target_style_tensors = [target_styles[i] for i in target_idx]
                if target_style_tensors:
                    target_style_tensor = torch.stack(target_style_tensors)  # [M, 10]
                    pred_style_matched = pred_style[b, pred_idx]  # [M, 10]
                    loss_style += F.mse_loss(pred_style_matched, target_style_tensor)

            # Hierarchy loss - predict parent relations
            if pred_hierarchy is not None and len(target_parent_ids) > 1:
                # Build parent relation matrix for matched queries
                # pred_hierarchy: [N, N] - probability that i is parent of j
                # We need to map parent_ids to matched query indices
                parent_id_to_query_idx = {}
                for q_idx, p_id in zip(pred_idx, target_parent_ids):
                    if p_id is not None:
                        parent_id_to_query_idx[p_id] = q_idx.item()

                # For each matched query, find its parent among matched queries
                for q_idx, p_id in zip(pred_idx, target_parent_ids):
                    if p_id is not None and p_id in parent_id_to_query_idx:
                        parent_q_idx = parent_id_to_query_idx[p_id]
                        # pred_hierarchy[parent, child] should be high
                        loss_hierarchy += F.binary_cross_entropy(
                            pred_hierarchy[b, parent_q_idx, q_idx],
                            torch.tensor(1.0, device=pred_hierarchy.device)
                        )

            # Unmatched predictions -> background
            all_pred_idx = torch.arange(N, device=pred_logits.device)
            unmatched = all_pred_idx[~torch.isin(all_pred_idx, pred_idx)]

            if len(unmatched) > 0:
                loss_type += F.cross_entropy(
                    pred_logits[b, unmatched],
                    torch.full((len(unmatched),), self.num_classes, dtype=torch.long, device=pred_logits.device)
                )
                loss_objectness += F.binary_cross_entropy(
                    pred_obj[b, unmatched],
                    torch.zeros(len(unmatched), device=pred_obj.device)
                )

        norm = max(num_matched, 1)
        loss_type = loss_type / B
        loss_bbox = loss_bbox / max(num_matched, 1)
        loss_giou = loss_giou / max(num_matched, 1)
        loss_objectness = loss_objectness / B
        loss_style = loss_style / max(num_matched, 1)
        loss_hierarchy = loss_hierarchy / max(num_matched, 1)

        # CLIP alignment loss
        student_clip = student_out.get('clip_proj', None)
        if student_clip is not None:
            loss_clip = F.mse_loss(student_clip, teacher_out['clip_embeddings'])
        else:
            student_clip = self.clip_proj(student_tokens.mean(dim=1))
            loss_clip = F.mse_loss(student_clip, teacher_out['clip_embeddings'])

        total = (self.lambda_type * loss_type +
                 self.lambda_bbox * loss_bbox +
                 self.lambda_giou * loss_giou +
                 self.lambda_objectness * loss_objectness +
                 self.lambda_style * loss_style +
                 self.lambda_hierarchy * loss_hierarchy +
                 self.lambda_clip * loss_clip)

        return {
            'type': loss_type,
            'bbox': loss_bbox,
            'giou': loss_giou,
            'objectness': loss_objectness,
            'style': loss_style,
            'hierarchy': loss_hierarchy,
            'clip': loss_clip,
            'total': total,
            'num_matched': num_matched
        }


def create_gate2_loss(config: dict) -> Gate2Loss:
    return Gate2Loss(
        num_classes=config.get('num_ui_classes', 15),
        lambda_type=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        lambda_objectness=1.0,
        lambda_style=1.0,
        lambda_hierarchy=1.0,
        lambda_clip=1.0
    )