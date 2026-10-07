#!/usr/bin/env python3
"""Gate 1 Overfit Proof - trains on 64 COCO images, evaluates WHAT + WHERE"""
import torch
import torch.nn as nn
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from earai.training.gate1_student import create_gate1_student
from earai.training.gate1_teachers import create_gate1_teachers
from earai.training.gate1_loss import create_gate1_loss
from earai.training.gate1_dataset import create_gate1_dataloader
from earai.training.gate1_decoder import create_gate1_decoder
from earai.training.gate1_cache import build_teacher_cache, create_cached_dataloader


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes"""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


def compute_metrics(pred_boxes: torch.Tensor, pred_labels: torch.Tensor, pred_obj: torch.Tensor,
                   target_boxes: List[torch.Tensor], target_labels: List[torch.Tensor],
                   threshold: float = 0.5) -> Dict:
    """
    Compute precision, recall, F1, mean IoU, class accuracy
    """
    all_ious = []
    tp = 0
    fp = 0
    fn = 0
    correct_class = 0
    total_class = 0

    for b in range(len(target_boxes)):
        gt_boxes = target_boxes[b]
        gt_labels = target_labels[b]
        num_gt = len(gt_boxes)

        # Get predictions above threshold
        pred_mask = pred_obj[b] >= threshold
        pred_boxes_b = pred_boxes[b][pred_mask]
        pred_labels_b = pred_labels[b][pred_mask]
        pred_obj_b = pred_obj[b][pred_mask]
        num_pred = len(pred_boxes_b)

        if num_gt == 0:
            # No GT objects - all predictions are FP
            fp += num_pred
            continue

        if num_pred == 0:
            # No predictions - all GT are FN
            fn += num_gt
            continue

        # IoU matrix
        ious = box_iou(pred_boxes_b, gt_boxes)  # [P, G]
        ious_np = ious.detach().cpu().numpy()

        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        cost = -ious_np
        pred_idx, target_idx = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()

        for p_idx, t_idx in zip(pred_idx, target_idx):
            if p_idx < num_pred and t_idx < num_gt:
                iou = ious[p_idx, t_idx].item()
                # Always mark as matched (to avoid double-counting)
                matched_gt.add(t_idx)
                matched_pred.add(p_idx)

                # Only count as TP if IoU >= 0.5
                if iou >= 0.5:
                    tp += 1
                    all_ious.append(iou)

                    # Class accuracy on matched
                    if pred_labels_b[p_idx].item() == gt_labels[t_idx].item():
                        correct_class += 1
                    total_class += 1
                else:
                    # Low IoU = FP + FN (but already marked as matched)
                    fp += 1
                    fn += 1

        # Unmatched predictions = FP
        fp += num_pred - len(matched_pred)
        # Unmatched GT = FN
        fn += num_gt - len(matched_gt)

    # Compute metrics
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    class_acc = correct_class / max(total_class, 1)

    return {
        'mean_iou': float(mean_iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_accuracy': float(class_acc),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


class Gate1Trainer:
    """Gate 1 Trainer with teacher caching"""

    def __init__(self, config: dict, device: str = 'cpu', teacher_cache: Dict = None):
        self.config = config
        self.device = torch.device(device)
        self.teacher_cache = teacher_cache

        # Models
        print("Creating student...")
        self.student = create_gate1_student(config).to(self.device)
        print(f"Student params: {sum(p.numel() for p in self.student.parameters())/1e6:.2f}M")

        print("Creating decoder...")
        self.decoder = create_gate1_decoder(config).to(self.device)
        print(f"Decoder params: {sum(p.numel() for p in self.decoder.parameters())/1e6:.2f}M")

        # Loss (no clip_proj - it's in decoder now)
        print("Creating loss...")
        self.loss_fn = create_gate1_loss(config).to(self.device)

        # Optimizer - INCLUDE decoder.clip_proj
        print("Creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + 
            list(self.decoder.parameters()),
            lr=config.get('lr', 0.0001),
            weight_decay=config.get('weight_decay', 0.0001)
        )

        # Dataloader - use cached teacher targets
        if teacher_cache is not None:
            print("Creating cached dataloader...")
            from earai.training.gate1_cache import create_cached_dataloader
            self.train_loader = create_cached_dataloader(config, teacher_cache, self.device, shuffle=True)
        else:
            print("Creating dataloader (no cache)...")
            from earai.training.gate1_dataset import create_gate1_dataloader
            self.train_loader = create_gate1_dataloader(config, shuffle=True)

        # Metrics tracking
        self.history = []
        self.initial_loss = None
        self.best_f1 = 0
        self.best_state = None

    def train_step(self, batch: Dict) -> Dict:
        """Single training step"""
        self.student.train()
        self.decoder.train()
        self.loss_fn.train()

        student_imgs = batch['student'].to(self.device)

        # Teacher targets are already in batch (cached)
        teacher_out = {
            'detections': [
                {'boxes': b.to(self.device), 'labels': l.to(self.device), 'scores': torch.ones_like(l, dtype=torch.float32)}
                for b, l in zip(batch['boxes'], batch['labels'])
            ],
            'clip_embeddings': batch['clip_embeddings'].to(self.device)
        }

        # Student forward
        student_tokens = self.student(student_imgs)
        student_out = self.decoder(student_tokens)

        # Loss
        losses = self.loss_fn(student_tokens, student_out, teacher_out)

        # Backward
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.student.parameters()) + 
            list(self.decoder.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        # Convert to scalars
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, save_predictions: bool = False):
        """Evaluate on training set (overfit test)"""
        self.student.eval()
        self.decoder.eval()

        all_pred_boxes = []
        all_pred_labels = []
        all_pred_obj = []
        all_target_boxes = []
        all_target_labels = []
        all_image_ids = []

        predictions_list = []

        for batch in self.train_loader:
            # Teacher targets from cache
            teacher_out = {
                'detections': [
                    {'boxes': b.to(self.device), 'labels': l.to(self.device), 'scores': torch.ones_like(l, dtype=torch.float32)}
                    for b, l in zip(batch['boxes'], batch['labels'])
                ],
                'clip_embeddings': batch['clip_embeddings'].to(self.device)
            }

            # Student predictions
            student_tokens = self.student(batch['student'].to(self.device))
            student_out = self.decoder(student_tokens)

            pred_boxes = student_out['bboxes_xyxy']     # [B, 16, 4]
            pred_labels = student_out['class_logits'].argmax(-1)  # [B, 16]
            pred_obj = student_out['objectness']        # [B, 16]

            # Targets from cache
            target_boxes = [b.to(self.device) for b in batch['boxes']]
            target_labels = [l.to(self.device) for l in batch['labels']]

            all_pred_boxes.append(pred_boxes)
            all_pred_labels.append(pred_labels)
            all_pred_obj.append(pred_obj)
            all_target_boxes.extend(target_boxes)
            all_target_labels.extend(target_labels)
            all_image_ids.extend(batch['image_ids'])

            if save_predictions:
                # Collect predictions for JSON
                for i in range(len(batch['image_ids'])):
                    img_id = batch['image_ids'][i]

                    # Teacher detections from cache
                    teacher_det = {
                        'boxes': batch['boxes'][i],
                        'labels': batch['labels'][i],
                        'scores': torch.ones_like(batch['labels'][i], dtype=torch.float32)
                    }
                    teacher_objects = []
                    for box, label, score in zip(teacher_det['boxes'], teacher_det['labels'], teacher_det['scores']):
                        if score >= 0.7:
                            teacher_objects.append({
                                'class': COCO_CATEGORIES[label] if label < len(COCO_CATEGORIES) else 'unknown',
                                'bbox': box.tolist(),
                                'confidence': float(score)
                            })

                    # EarAI predictions (threshold 0.5)
                    pred_mask = pred_obj[i] >= 0.5
                    earai_objects = []
                    for box, label, obj_score in zip(pred_boxes[i][pred_mask], pred_labels[i][pred_mask], pred_obj[i][pred_mask]):
                        earai_objects.append({
                            'class': COCO_CATEGORIES[label.item()] if label.item() < len(COCO_CATEGORIES) else 'unknown',
                            'bbox': box.tolist(),
                            'confidence': float(obj_score.item())
                        })

                    predictions_list.append({
                        'image_id': int(img_id),
                        'teacher': teacher_objects,
                        'earai': earai_objects
                    })

        # Concat all predictions
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_labels = torch.cat(all_pred_labels, dim=0)
        all_pred_obj = torch.cat(all_pred_obj, dim=0)

        # Compute metrics
        metrics = compute_metrics(
            all_pred_boxes, all_pred_labels, all_pred_obj,
            all_target_boxes, all_target_labels
        )

        if save_predictions:
            return metrics, predictions_list

        return metrics

    def train_epoch(self) -> Dict:
        """Train one epoch"""
        epoch_losses = {}

        for batch in tqdm(self.train_loader, desc='Training'):
            losses = self.train_step(batch)

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v)

        # Average
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        if self.initial_loss is None:
            self.initial_loss = avg_losses['total']

        self.history.append(avg_losses)
        return avg_losses

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'student_state': self.student.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'loss_fn_state': self.loss_fn.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt['student_state'])
        self.decoder.load_state_dict(ckpt['decoder_state'])
        self.loss_fn.load_state_dict(ckpt['loss_fn_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.history = ckpt.get('history', [])


def run_gate1_overfit(config: dict, device: str = 'auto') -> Dict:
    """Run Gate 1 overfit experiment with teacher caching"""

    # Auto-detect device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print("=" * 60)
    print("GATE 1 OVERFIT PROOF")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch_size', 8)}")
    print()

    # Build teacher cache (one-time)
    print("Building/loading teacher cache...")
    teacher_cache = build_teacher_cache(config, device='cpu')  # Teachers on CPU

    # Update config to use cached dataloader
    config['device'] = device

    trainer = Gate1Trainer(config, device, teacher_cache)

    epochs = config.get('epochs', 100)
    eval_every = config.get('eval_every', 10)

    for epoch in range(epochs):
        # Train
        train_losses = trainer.train_epoch()
        print(f"\nEpoch {epoch}: {train_losses}")

        # Evaluate
        if epoch % eval_every == 0 or epoch == epochs - 1:
            print("Evaluating...")
            metrics = trainer.evaluate()
            print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  Class Acc: {metrics['class_accuracy']:.4f}")

            if metrics['f1'] > trainer.best_f1:
                trainer.best_f1 = metrics['f1']
                trainer.best_state = {
                    'student': trainer.student.state_dict(),
                    'decoder': trainer.decoder.state_dict(),
                    'loss_fn': trainer.loss_fn.state_dict(),
                    'optimizer': trainer.optimizer.state_dict()
                }

    # Final evaluation with predictions
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_metrics, predictions = trainer.evaluate(save_predictions=True)
    print(f"Initial Loss: {trainer.initial_loss:.4f}")
    print(f"Final Loss: {trainer.history[-1]['total']:.4f}")
    print(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1: {final_metrics['f1']:.4f}")
    print(f"Class Accuracy: {final_metrics['class_accuracy']:.4f}")

    # PASS/FAIL criteria (updated)
    pass_gate = (
        final_metrics['mean_iou'] >= 0.70 and
        final_metrics['class_accuracy'] >= 0.90 and
        final_metrics['precision'] >= 0.90 and
        final_metrics['recall'] >= 0.90
    )

    result = {
        'initial_loss': trainer.initial_loss,
        'final_loss': trainer.history[-1]['total'],
        'mean_iou': final_metrics['mean_iou'],
        'class_accuracy': final_metrics['class_accuracy'],
        'precision': final_metrics['precision'],
        'recall': final_metrics['recall'],
        'f1': final_metrics['f1'],
        'steps': len(trainer.history) * len(trainer.train_loader),
        'epochs': epochs,
        'PASS': pass_gate
    }

    # Save artifacts
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)

    with open(artifacts_dir / 'gate1_report.json', 'w') as f:
        json.dump(result, f, indent=2)

    with open(artifacts_dir / 'gate1_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)

    # Save best checkpoint
    if trainer.best_state:
        torch.save(trainer.best_state, artifacts_dir / 'gate1_best.pt')

    print(f"\nResult: {'PASS \u2713' if pass_gate else 'FAIL \u2717'}")
    print(f"Report saved to: {artifacts_dir / 'gate1_report.json'}")
    print(f"Predictions saved to: {artifacts_dir / 'gate1_predictions.json'}")
    print(f"Best checkpoint saved to: {artifacts_dir / 'gate1_best.pt'}")

    return result


# COCO categories for prediction saving
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gate1.yaml')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config['epochs'] = args.epochs
    config['device'] = args.device
    config['eval_every'] = 10

    result = run_gate1_overfit(config, args.device)

    # Print summary for the user
    print("\n" + "=" * 60)
    print("GATE 1 RESULT SUMMARY")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  gate1_report.json: artifacts/gate1_report.json")
    print(f"  gate1_predictions.json: artifacts/gate1_predictions.json")
    print(f"  gate1_best.pt: artifacts/gate1_best.pt")
    print("=" * 60)