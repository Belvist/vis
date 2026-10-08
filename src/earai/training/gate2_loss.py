"""Gate 2 losses for screenshot -> UI structure learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]

    ilt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    irb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    iwh = (irb - ilt).clamp(min=0)
    inter = iwh[..., 0] * iwh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return iou - (area_c - union) / (area_c + 1e-6)


def _as_int(value) -> int:
    return int(value.item()) if torch.is_tensor(value) else int(value)


def _as_box(value, device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(value, device=device, dtype=torch.float32)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        results = []
        for b in range(pred_logits.shape[0]):
            elems = targets[b]["ui_elements"]
            if not elems:
                empty = torch.empty(0, dtype=torch.long, device=pred_logits.device)
                results.append((empty, empty))
                continue
            target_boxes = torch.stack([_as_box(e["bbox"], pred_boxes.device) for e in elems])
            target_labels = torch.tensor(
                [_as_int(e["class_id"]) for e in elems],
                dtype=torch.long, device=pred_logits.device,
            )

            prob = pred_logits[b].softmax(-1)
            class_cost = -prob[:, target_labels]
            bbox_cost = torch.cdist(pred_boxes[b], target_boxes, p=1)
            giou_cost = 1.0 - generalized_box_iou(pred_boxes[b], target_boxes)
            cost = (self.cost_class * class_cost +
                    self.cost_bbox * bbox_cost +
                    self.cost_giou * giou_cost)
            pi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
            results.append((
                torch.tensor(pi, dtype=torch.long, device=pred_logits.device),
                torch.tensor(ti, dtype=torch.long, device=pred_logits.device),
            ))
        return results


def focal_cross_entropy(logits, targets, alpha=None, gamma=2.0, reduction='mean'):
    """Focal loss for class imbalance."""
    # If alpha provided, ensure it has correct length (num_classes + 1 for background)
    if alpha is not None:
        num_classes = logits.shape[-1]
        if len(alpha) != num_classes:
            # Pad with 1.0 for background class
            alpha = torch.cat([alpha, torch.ones(1, device=alpha.device, dtype=alpha.dtype)])
    ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def focal_bce_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """Focal BCE for objectness (handles class imbalance between obj/background)."""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce * target + (1 - alpha) * pt ** gamma * bce * (1 - target)
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    return focal


class Gate2Loss(nn.Module):
    def __init__(self, num_classes=15, style_dim=10,
                 lambda_type=1.0, lambda_bbox=5.0, lambda_giou=2.0,
                 lambda_objectness=1.0, lambda_style=1.0,
                 lambda_hierarchy=1.0, lambda_clip=1.0,
                 class_weights=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.style_dim = int(style_dim)
        self.lambda_type = float(lambda_type)
        self.lambda_bbox = float(lambda_bbox)
        self.lambda_giou = float(lambda_giou)
        self.lambda_objectness = float(lambda_objectness)
        self.lambda_style = float(lambda_style)
        self.lambda_hierarchy = float(lambda_hierarchy)
        self.lambda_clip = float(lambda_clip)
        
        # Class weights for imbalanced classes (register as buffer)
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
            
        self.matcher = HungarianMatcher(lambda_type, lambda_bbox, lambda_giou)

    def forward(self, student_tokens, student_out, teacher_out):
        logits = student_out["class_logits"]
        boxes = student_out["bboxes_xyxy"]
        obj = student_out["objectness"]
        styles = student_out["style"]
        hierarchy = student_out["hierarchy"]
        targets = teacher_out["targets"]

        B, N, _ = logits.shape
        matches = self.matcher(logits, boxes, targets)

        loss_type = torch.tensor(0.0, device=logits.device)
        loss_obj = torch.tensor(0.0, device=logits.device)
        loss_bbox = torch.tensor(0.0, device=logits.device)
        loss_giou = torch.tensor(0.0, device=logits.device)
        loss_style = torch.tensor(0.0, device=logits.device)
        loss_hierarchy = torch.tensor(0.0, device=logits.device)

        matched_total = 0
        hierarchy_batches = 0

        for b, (pred_idx, target_idx) in enumerate(matches):
            elems = targets[b]["ui_elements"]

            class_targets = torch.full(
                (N,), self.num_classes, dtype=torch.long, device=logits.device
            )
            obj_targets = torch.zeros(N, dtype=torch.float32, device=obj.device)

            if len(pred_idx):
                matched_elems = [elems[int(i)] for i in target_idx.tolist()]
                target_labels = torch.tensor(
                    [_as_int(e["class_id"]) for e in matched_elems],
                    dtype=torch.long, device=logits.device,
                )
                target_boxes = torch.stack(
                    [_as_box(e["bbox"], boxes.device) for e in matched_elems]
                )
                class_targets[pred_idx] = target_labels
                obj_targets[pred_idx] = 1.0

                loss_bbox = loss_bbox + F.l1_loss(
                    boxes[b, pred_idx], target_boxes, reduction="sum"
                )
                giou_diag = torch.diag(
                    generalized_box_iou(boxes[b, pred_idx], target_boxes)
                )
                loss_giou = loss_giou + (1.0 - giou_diag).sum()

                target_style = torch.stack([
                    e["style"].to(styles.device, dtype=torch.float32)
                    if torch.is_tensor(e["style"])
                    else torch.tensor(e["style"], device=styles.device, dtype=torch.float32)
                    for e in matched_elems
                ])
                if target_style.shape[-1] != self.style_dim:
                    raise RuntimeError(
                        f"Gate2 style target dim {target_style.shape[-1]} != {self.style_dim}"
                    )
                loss_style = loss_style + F.mse_loss(
                    styles[b, pred_idx], target_style, reduction="sum"
                )
                matched_total += len(pred_idx)

                # Train parent->child relations with both positive and negative pairs.
                if len(pred_idx) > 1:
                    id_to_local = {
                        e.get("element_id"): i for i, e in enumerate(matched_elems)
                        if e.get("element_id") is not None
                    }
                    target_rel = torch.zeros(
                        (len(pred_idx), len(pred_idx)),
                        device=hierarchy.device, dtype=torch.float32,
                    )
                    for child_i, e in enumerate(matched_elems):
                        parent_id = e.get("parent_id")
                        if parent_id in id_to_local:
                            target_rel[id_to_local[parent_id], child_i] = 1.0

                    pred_rel = hierarchy[b].index_select(0, pred_idx).index_select(1, pred_idx)
                    off_diag = ~torch.eye(len(pred_idx), device=hierarchy.device, dtype=torch.bool)
                    pos = off_diag & (target_rel > 0.5)
                    neg = off_diag & ~pos
                    rel_loss = torch.tensor(0.0, device=hierarchy.device)
                    if pos.any():
                        rel_loss = rel_loss + F.binary_cross_entropy(
                            pred_rel[pos], torch.ones_like(pred_rel[pos])
                        )
                    if neg.any():
                        rel_loss = rel_loss + 0.25 * F.binary_cross_entropy(
                            pred_rel[neg], torch.zeros_like(pred_rel[neg])
                        )
                    loss_hierarchy = loss_hierarchy + rel_loss
                    hierarchy_batches += 1

            # Standard CE with class weights for imbalance
            class_w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            if class_w is not None and class_w.shape[0] != logits.shape[-1]:
                # Pad with 1.0 for background class
                class_w = torch.cat([class_w, torch.ones(1, device=class_w.device, dtype=class_w.dtype)])
            loss_type = loss_type + F.cross_entropy(logits[b], class_targets, weight=class_w)
            
            # BCE for objectness - matched=1, unmatched=0 (background)
            loss_obj = loss_obj + F.binary_cross_entropy(obj[b], obj_targets)

        matched_norm = max(matched_total, 1)
        loss_type = loss_type / B
        loss_obj = loss_obj / B
        loss_bbox = loss_bbox / matched_norm
        loss_giou = loss_giou / matched_norm
        loss_style = loss_style / (matched_norm * self.style_dim)
        loss_hierarchy = loss_hierarchy / max(hierarchy_batches, 1)

        student_clip = F.normalize(student_out["clip_proj"], dim=-1)
        teacher_clip = F.normalize(
            teacher_out["clip_embeddings"].to(student_clip.device), dim=-1
        )
        loss_clip = (1.0 - (student_clip * teacher_clip).sum(dim=-1)).mean()

        total = (
            self.lambda_type * loss_type +
            self.lambda_bbox * loss_bbox +
            self.lambda_giou * loss_giou +
            self.lambda_objectness * loss_obj +
            self.lambda_style * loss_style +
            self.lambda_hierarchy * loss_hierarchy +
            self.lambda_clip * loss_clip
        )
        return {
            "type": loss_type,
            "bbox": loss_bbox,
            "giou": loss_giou,
            "objectness": loss_obj,
            "style": loss_style,
            "hierarchy": loss_hierarchy,
            "clip": loss_clip,
            "total": total,
            "num_matched": matched_total,
        }


def create_gate2_loss(config: dict) -> Gate2Loss:
    # Default class weights based on diagnostics (inverse frequency)
    class_weights = config.get("class_weights")
    return Gate2Loss(
        num_classes=int(config.get("num_ui_classes", 15)),
        style_dim=int(config.get("style_dim", 10)),
        lambda_type=config.get("weight_type", 1.0),
        lambda_bbox=config.get("weight_bbox", 5.0),
        lambda_giou=config.get("weight_giou", 2.0),
        lambda_objectness=config.get("weight_objectness", 1.0),
        lambda_style=config.get("weight_style", 1.0),
        lambda_hierarchy=config.get("weight_hierarchy", 1.0),
        lambda_clip=config.get("weight_clip", 1.0),
        class_weights=class_weights,
    )