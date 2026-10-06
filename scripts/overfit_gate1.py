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
import sys

sys.path.insert(0, '/Users/earflow/earai/src')

from earai.training.student import create_student_model
from earai.training.gate1_teachers import create_gate1_teachers
from earai.training.gate1_loss import create_gate1_loss
from earai.training.gate1_dataset import create_gate1_dataloader
from earai.decoder.heads import create_decoder


def compute_metrics(pred_boxes: torch.Tensor, pred_labels: torch.Tensor, pred_obj: torch.Tensor,
                   target_boxes: List[torch.Tensor], target_labels: List[torch.Tensor],
                   threshold: float = 0.5) -> Dict:
    """Compute IoU, class accuracy, objectness accuracy"""
    
    all_ious = []
    correct_class = 0
    total_class = 0
    correct_obj = 0
    total_obj = 0
    
    for b in range(len(target_boxes)):
        if len(target_boxes[b]) == 0:
            # No GT objects - all predictions should be background
            pred_obj_b = pred_obj[b]
            correct_obj += (pred_obj_b < threshold).sum().item()
            total_obj += len(pred_obj_b)
            continue
        
        # Get predictions above threshold
        pred_mask = pred_obj[b] >= threshold
        pred_boxes_b = pred_boxes[b][pred_mask]
        pred_labels_b = pred_labels[b][pred_mask]
        pred_obj_b = pred_obj[b][pred_mask]
        
        if len(pred_boxes_b) == 0:
            # No predictions above threshold
            total_obj += len(target_boxes[b])
            continue
        
        # IoU with Hungarian matching
        from scipy.optimize import linear_sum_assignment
        
        ious = box_iou(pred_boxes_b, target_boxes[b])  # [P, G]
        ious_np = ious.detach().cpu().numpy()
        
        # Hungarian assignment
        cost = -ious_np
        pred_idx, target_idx = linear_sum_assignment(cost)
        
        # Matched pairs
        for p_idx, t_idx in zip(pred_idx, target_idx):
            if p_idx < len(pred_boxes_b) and t_idx < len(target_boxes[b]):
                iou = ious[p_idx, t_idx].item()
                all_ious.append(iou)
                
                # Class accuracy
                if pred_labels_b[p_idx].item() == target_labels[b][t_idx].item():
                    correct_class += 1
                total_class += 1
        
        # Unmatched predictions = false positives
        matched_pred = set(pred_idx)
        for i in range(len(pred_boxes_b)):
            if i not in matched_pred:
                total_obj += 1  # False positive
        
        # Unmatched targets = false negatives
        matched_target = set(target_idx)
        for i in range(len(target_boxes[b])):
            if i not in matched_target:
                total_obj += 1  # False negative
        
        # Matched = true positives
        correct_obj += len(matched_pred)
    
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    class_acc = correct_class / max(total_class, 1)
    obj_acc = correct_obj / max(total_obj, 1)
    
    return {
        'mean_iou': float(mean_iou),
        'class_accuracy': float(class_acc),
        'objectness_accuracy': float(obj_acc),
        'num_matched': len(all_ious)
    }


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


class Gate1Trainer:
    """Gate 1 Trainer with proper optimizer including clip_proj"""
    
    def __init__(self, config: dict, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        
        # Models
        print("Creating student...")
        self.student = create_student_model(config).to(self.device)
        print(f"Student params: {sum(p.numel() for p in self.student.parameters())/1e6:.2f}M")
        
        print("Creating decoder...")
        self.decoder = create_decoder(config).to(self.device)
        print(f"Decoder params: {sum(p.numel() for p in self.decoder.parameters())/1e6:.2f}M")
        
        print("Loading teachers...")
        self.teachers, _ = create_gate1_teachers(device)
        
        # Loss (includes clip_proj)
        print("Creating loss...")
        self.loss_fn = create_gate1_loss(config).to(self.device)
        
        # Optimizer - INCLUDE loss_fn.clip_proj
        print("Creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.loss_fn.clip_proj.parameters()),
            lr=config.get('lr', 0.0001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # Dataloader
        print("Creating dataloader...")
        self.train_loader = create_gate1_dataloader(config, shuffle=True)
        
        # Metrics tracking
        self.history = []
        self.initial_loss = None
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step"""
        self.student.train()
        self.decoder.train()
        self.loss_fn.train()
        
        student_imgs = batch['student'].to(self.device)
        raw_imgs = batch['raw'].to(self.device)
        clip_imgs = batch['clip'].to(self.device)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teachers(student_imgs, raw_imgs, clip_imgs)
        
        # Student forward
        student_tokens = self.student(student_imgs)  # [B, 16, 256]
        student_out = self.decoder(student_tokens)
        
        # Loss
        losses = self.loss_fn(student_tokens, student_out, teacher_out)
        
        # Backward
        self.optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.student.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.loss_fn.clip_proj.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Convert to scalars
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on training set (overfit test)"""
        self.student.eval()
        self.decoder.eval()
        
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_obj = []
        all_target_boxes = []
        all_target_labels = []
        
        for batch in self.train_loader:
            student_imgs = batch['student'].to(self.device)
            raw_imgs = batch['raw'].to(self.device)
            clip_imgs = batch['clip'].to(self.device)
            
            # Teacher targets
            with torch.no_grad():
                teacher_out = self.teachers(student_imgs, raw_imgs, clip_imgs)
            
            # Student predictions
            student_tokens = self.student(student_imgs)
            student_out = self.decoder(student_tokens)
            
            pred_boxes = student_out['bboxes_xyxy']     # [B, 16, 4]
            pred_labels = student_out['class_logits'].argmax(-1)  # [B, 16]
            pred_obj = student_out['objectness']        # [B, 16]
            
            # Targets
            target_boxes = [b.to(self.device) for b in batch['boxes']]
            target_labels = [l.to(self.device) for l in batch['labels']]
            
            all_pred_boxes.append(pred_boxes)
            all_pred_labels.append(pred_labels)
            all_pred_obj.append(pred_obj)
            all_target_boxes.extend(target_boxes)
            all_target_labels.extend(target_labels)
        
        # Concat all predictions
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_labels = torch.cat(all_pred_labels, dim=0)
        all_pred_obj = torch.cat(all_pred_obj, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(
            all_pred_boxes, all_pred_labels, all_pred_obj,
            all_target_boxes, all_target_labels
        )
        
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


def run_gate1_overfit(config: dict, device: str = 'cpu') -> Dict:
    """Run Gate 1 overfit experiment"""
    
    print("=" * 60)
    print("GATE 1 OVERFIT PROOF")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch_size', 16)}")
    print()
    
    trainer = Gate1Trainer(config, device)
    
    epochs = config.get('epochs', 100)
    eval_every = config.get('eval_every', 10)
    
    best_iou = 0
    final_metrics = None
    
    for epoch in range(epochs):
        # Train
        train_losses = trainer.train_epoch()
        print(f"\nEpoch {epoch}: {train_losses}")
        
        # Evaluate
        if epoch % eval_every == 0 or epoch == epochs - 1:
            print("Evaluating...")
            metrics = trainer.evaluate()
            print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Class Acc: {metrics['class_accuracy']:.4f}")
            print(f"  Obj Acc: {metrics['objectness_accuracy']:.4f}")
            
            if metrics['mean_iou'] > best_iou:
                best_iou = metrics['mean_iou']
            final_metrics = metrics
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_metrics = trainer.evaluate()
    print(f"Initial Loss: {trainer.initial_loss:.4f}")
    print(f"Final Loss: {trainer.history[-1]['total']:.4f}")
    print(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
    print(f"Class Accuracy: {final_metrics['class_accuracy']:.4f}")
    print(f"Objectness Accuracy: {final_metrics['objectness_accuracy']:.4f}")
    
    # PASS/FAIL criteria
    pass_gate = (
        final_metrics['mean_iou'] >= 0.70 and
        final_metrics['class_accuracy'] >= 0.90 and
        final_metrics['objectness_accuracy'] >= 0.95
    )
    
    result = {
        'initial_loss': trainer.initial_loss,
        'final_loss': trainer.history[-1]['total'],
        'mean_iou': final_metrics['mean_iou'],
        'class_accuracy': final_metrics['class_accuracy'],
        'objectness_accuracy': final_metrics['objectness_accuracy'],
        'steps': len(trainer.history) * len(trainer.train_loader),
        'epochs': epochs,
        'PASS': pass_gate
    }
    
    # Save report
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    with open(artifacts_dir / 'gate1_report.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save predictions for inspection
    # (would need one more forward pass to collect)
    
    print(f"\nResult: {'PASS ✓' if pass_gate else 'FAIL ✗'}")
    print(f"Report saved to: {artifacts_dir / 'gate1_report.json'}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--device', type=str, default='cpu')
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
    print("=" * 60)