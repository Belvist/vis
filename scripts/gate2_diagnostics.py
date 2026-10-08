#!/usr/bin/env python3
"""
Diagnostic evaluation for Gate 2 - runs on existing checkpoint without retraining.
Outputs: artifacts/gate2_diagnostics.json
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.earai.training.gate2_student import create_gate2_student
from src.earai.training.gate2_cache import build_teacher_cache, create_cached_dataloader
from src.earai.training.gate2_loss import HungarianMatcher, Gate2Loss
import yaml

# Config
CHECKPOINT = Path("artifacts/gate2_best.pt")
CONFIG = Path("configs/gate2.yaml")
OUTPUT = Path("artifacts/gate2_diagnostics.json")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load config
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

# Load model
model = create_gate2_student(cfg).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["student_state"])
model.eval()

# Build cache (or load existing)
cache = build_teacher_cache(cfg, device=str(DEVICE))

# Data loaders
val_loader = create_cached_dataloader(cfg, cache, device=str(DEVICE), shuffle=False, split="val", drop_last=False)
train_loader = create_cached_dataloader(cfg, cache, device=str(DEVICE), shuffle=False, split="train", drop_last=False)

# Loss & matcher
matcher = HungarianMatcher(
    cost_class=cfg.get("cost_class", 1.0),
    cost_bbox=cfg.get("cost_bbox", 5.0),
    cost_giou=cfg.get("cost_giou", 2.0),
)
criterion = Gate2Loss(
    num_classes=cfg["num_ui_classes"],
    style_dim=cfg["style_dim"],
    lambda_type=cfg.get("weight_type", 1.0),
    lambda_bbox=cfg.get("weight_bbox", 5.0),
    lambda_giou=cfg.get("weight_giou", 2.0),
    lambda_objectness=cfg.get("weight_objectness", 1.0),
    lambda_style=cfg.get("weight_style", 1.0),
    lambda_hierarchy=cfg.get("weight_hierarchy", 1.0),
    lambda_clip=cfg.get("weight_clip", 1.0),
).to(DEVICE)

CLASS_NAMES = [
    "navbar", "hero", "section", "container", "card", "button", "input",
    "image", "icon", "heading", "paragraph", "badge", "modal", "footer", "link"
]

print("Running diagnostics...")

# Helper: convert batch to matcher targets format
def batch_to_matcher_targets(batch, device):
    """Convert cached dataloader batch to HungarianMatcher targets format."""
    targets = []
    for i in range(len(batch["student"])):
        boxes = batch["boxes"][i].to(device)
        labels = batch["labels"][i].to(device)
        element_ids = batch["element_ids"][i]
        parent_ids = batch["parent_ids"][i]
        
        elems = []
        for j in range(len(boxes)):
            elems.append({
                "element_id": element_ids[j],
                "parent_id": parent_ids[j],
                "bbox": boxes[j],
                "class_id": int(labels[j].item()),
                "text": batch["texts"][i][j],
                "style": batch["styles"][i][j],
            })
        targets.append({"ui_elements": elems})
    return targets

# ============================================================
# 1. THRESHOLD SWEEP
# ============================================================
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
threshold_results = {}

print("Threshold sweep...")
with torch.no_grad():
    for thresh in thresholds:
        tp = fp = fn = 0
        active_queries = []
        
        for batch in val_loader:
            images = batch["student"].to(DEVICE)
            matcher_targets = batch_to_matcher_targets(batch, DEVICE)
            
            outputs = model(images)
            pred_logits = outputs["class_logits"]
            pred_boxes = outputs["bboxes_xyxy"]
            pred_objectness = outputs["objectness"].sigmoid()
            
            for b in range(len(images)):
                # Hungarian matching for this image
                matches = matcher(
                    pred_logits[b:b+1], pred_boxes[b:b+1], [matcher_targets[b]]
                )
                src_idx, tgt_idx = matches[0]
                
                elems = matcher_targets[b]["ui_elements"]
                num_gt = len([e for e in elems if e["class_id"] != 14])
                
                matched = set(src_idx.tolist())
                matched_list = list(matched)
                
                # Active queries at this threshold
                active = (pred_objectness[b] >= thresh).sum().item()
                active_queries.append(active)
                
                for q in range(len(pred_objectness[b])):
                    if pred_objectness[b, q] < thresh:
                        continue
                    if q in matched:
                        matched_idx = matched_list.index(q)
                        gt_class = elems[tgt_idx[matched_idx]]["class_id"]
                        if gt_class != 14:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        fp += 1
                
                # FN
                matched_active = [q for q in matched if pred_objectness[b, q] >= thresh]
                fn += num_gt - len(matched_active)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_results[str(thresh)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "avg_active_queries_per_image": round(np.mean(active_queries), 2),
        }

best_thresh = max(threshold_results, key=lambda k: threshold_results[k]["f1"])
best_f1 = threshold_results[best_thresh]["f1"]

# ============================================================
# 2. OBJECTNESS DISTRIBUTIONS
# ============================================================
print("Objectness distributions...")
pos_objectness = []
neg_objectness = []
all_active_counts = []
val_gt_counts = []

with torch.no_grad():
    for batch in val_loader:
        images = batch["student"].to(DEVICE)
        matcher_targets = batch_to_matcher_targets(batch, DEVICE)
        
        outputs = model(images)
        pred_logits = outputs["class_logits"]
        pred_boxes = outputs["bboxes_xyxy"]
        pred_objectness = outputs["objectness"].sigmoid()
        
        for b in range(len(images)):
            matches = matcher(
                pred_logits[b:b+1], pred_boxes[b:b+1], [matcher_targets[b]]
            )
            src_idx, tgt_idx = matches[0]
            matched = set(src_idx.tolist())
            matched_list = list(matched)
            
            elems = matcher_targets[b]["ui_elements"]
            val_gt_counts.append(len([e for e in elems if e["class_id"] != 14]))
            
            active = (pred_objectness[b] >= 0.5).sum().item()
            all_active_counts.append(active)
            
            for q in range(len(pred_objectness[b])):
                obj = pred_objectness[b, q].item()
                if q in matched:
                    mt = matched_list.index(q)
                    gt_class = elems[tgt_idx[mt]]["class_id"]
                    if gt_class != 14:
                        pos_objectness.append(obj)
                    else:
                        neg_objectness.append(obj)
                else:
                    neg_objectness.append(obj)

def stats(arr):
    arr = np.array(arr)
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }

pos_stats = stats(pos_objectness) if pos_objectness else {"mean": 0, "median": 0, "p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0}
neg_stats = stats(neg_objectness) if neg_objectness else {"mean": 0, "median": 0, "p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0}

# ============================================================
# 3. GT ELEMENT COUNT DISTRIBUTION (train + val)
# ============================================================
print("GT count distribution...")
train_gt_counts = []

with torch.no_grad():
    for batch in train_loader:
        for b in range(len(batch["student"])):
            labels = batch["labels"][b]
            train_gt_counts.append(len([l for l in labels if l != 14]))

def hist_counts(counts, name):
    arr = np.array(counts)
    bins = {
        "1-5": int(np.sum((arr >= 1) & (arr <= 5))),
        "6-10": int(np.sum((arr >= 6) & (arr <= 10))),
        "11-16": int(np.sum((arr >= 11) & (arr <= 16))),
        "17-24": int(np.sum((arr >= 17) & (arr <= 24))),
        "25-31": int(np.sum((arr >= 25) & (arr <= 31))),
        "32": int(np.sum(arr == 32)),
    }
    pct_truncated = bins["32"] / len(arr) * 100 if len(arr) > 0 else 0
    return {
        f"{name}_mean": round(float(arr.mean()), 2),
        f"{name}_median": round(float(np.median(arr)), 2),
        f"{name}_p90": round(float(np.percentile(arr, 90)), 2),
        f"{name}_max": int(arr.max()),
        f"{name}_histogram": bins,
        f"{name}_pct_truncated_at_32": round(pct_truncated, 2),
    }

gt_stats = {}
gt_stats.update(hist_counts(train_gt_counts, "train"))
gt_stats.update(hist_counts(val_gt_counts, "val"))

# ============================================================
# 4. PREDICTION COUNT DISTRIBUTION (threshold 0.5)
# ============================================================
active_arr = np.array(all_active_counts)
pred_stats = {
    "val_mean_active_queries": round(float(active_arr.mean()), 2),
    "val_median_active_queries": round(float(np.median(active_arr)), 2),
    "val_p90_active_queries": round(float(np.percentile(active_arr, 90)), 2),
}

# ============================================================
# 5. CLASS CONFUSION MATRIX
# ============================================================
print("Class confusion matrix...")
confusion = np.zeros((15, 15), dtype=int)  # 15 classes including link
class_support = Counter()
class_correct = Counter()
class_total_pred = Counter()

with torch.no_grad():
    for batch in val_loader:
        images = batch["student"].to(DEVICE)
        matcher_targets = batch_to_matcher_targets(batch, DEVICE)
        
        outputs = model(images)
        pred_logits = outputs["class_logits"]
        pred_boxes = outputs["bboxes_xyxy"]
        pred_objectness = outputs["objectness"].sigmoid()
        
        for b in range(len(images)):
            matches = matcher(
                pred_logits[b:b+1], pred_boxes[b:b+1], [matcher_targets[b]]
            )
            src_idx, tgt_idx = matches[0]
            
            elems = matcher_targets[b]["ui_elements"]
            
            for si, ti in zip(src_idx, tgt_idx):
                si = si.item()
                ti = ti.item()
                
                gt_class = elems[ti]["class_id"]
                # Include link class in confusion but not in per-class metrics
                
                pred_class = pred_logits[b, si].argmax().item()
                pred_obj = pred_objectness[b, si].item()
                
                if pred_obj >= 0.5:
                    # Bounds check
                    if gt_class < 15 and pred_class < 15:
                        confusion[gt_class, pred_class] += 1
                        class_support[gt_class] += 1
                        class_total_pred[pred_class] += 1
                        if gt_class == pred_class:
                            class_correct[gt_class] += 1

per_class = {}
for c in range(15):
    support = class_support.get(c, 0)
    correct = class_correct.get(c, 0)
    total_pred = class_total_pred.get(c, 0)
    per_class[CLASS_NAMES[c]] = {
        "support": support,
        "precision": round(correct / total_pred, 4) if total_pred > 0 else 0,
        "recall": round(correct / support, 4) if support > 0 else 0,
        "accuracy": round(correct / support, 4) if support > 0 else 0,
    }

top_confusions = []
for gt in range(15):
    for pred in range(15):
        if gt != pred and confusion[gt, pred] > 0:
            top_confusions.append({
                "gt": CLASS_NAMES[gt],
                "pred": CLASS_NAMES[pred],
                "count": int(confusion[gt, pred]),
            })
top_confusions.sort(key=lambda x: x["count"], reverse=True)
top_confusions = top_confusions[:20]

# ============================================================
# 6. DATASET CLASS BALANCE
# ============================================================
print("Class balance...")
train_class_counts = Counter()
val_class_counts = Counter()

with torch.no_grad():
    for loader, store in [(train_loader, train_class_counts), (val_loader, val_class_counts)]:
        for batch in loader:
            for b in range(len(batch["student"])):
                labels = batch["labels"][b]
                for l in labels:
                    store[l.item()] += 1

def class_dist(counts, total_name):
    total = sum(counts.values())
    dist = {}
    for c, cnt in counts.items():
        if c < len(CLASS_NAMES):
            dist[CLASS_NAMES[c]] = {
                "count": cnt,
                "percentage": round(cnt / total * 100, 2) if total > 0 else 0,
            }
    return {f"{total_name}_total": total, f"{total_name}_distribution": dist}

class_balance = {}
class_balance.update(class_dist(train_class_counts, "train"))
class_balance.update(class_dist(val_class_counts, "val"))

# ============================================================
# 7. FINAL DIAGNOSIS
# ============================================================
diag_notes = []

# A: Objectness calibration
best_f1_val = threshold_results[best_thresh]["f1"]
thresh_05_key = "0.5" if "0.5" in threshold_results else "0.50"
if best_f1_val > 0.5 and threshold_results[thresh_05_key]["f1"] < 0.2:
    diag_notes.append("A — objectness calibration problem (threshold fixes it)")

# B: Dataset/DOM label problem
max_conf = max(top_confusions, key=lambda x: x["count"]) if top_confusions else {"count": 0}
if max_conf["count"] > 100 and max_conf["gt"] in ["section", "container"] and max_conf["pred"] in ["section", "container"]:
    diag_notes.append("B — dataset/DOM label problem (section↔container confusion)")

# C: Class imbalance
train_dist = class_balance.get("train_distribution", {})
if train_dist:
    percentages = [v["percentage"] for v in train_dist.values()]
    min_pct = min([p for p in percentages if p > 0], default=1)
    if max(percentages) > 3 * min_pct:
        diag_notes.append("C — class imbalance problem")

# D: Query capacity
val_p95_gt = np.percentile(val_gt_counts, 95)
if val_p95_gt > 32:
    diag_notes.append("D — query capacity problem (p95 GT > 32)")
elif val_p95_gt > 20:
    diag_notes.append("D — query capacity pressure (p95 GT > 20)")

# E: Combination
if len(diag_notes) > 1:
    diag_notes = ["E — combination: " + "; ".join(diag_notes)]

diagnosis = diag_notes[0] if diag_notes else "Unknown"

# ============================================================
# OUTPUT
# ============================================================
output = {
    "best_threshold": float(best_thresh),
    "best_f1": round(best_f1, 4),
    "threshold_sweep": threshold_results,
    "objectness_positive_distribution": pos_stats,
    "objectness_background_distribution": neg_stats,
    "gt_count_distribution": gt_stats,
    "prediction_count_distribution": pred_stats,
    "class_confusion_matrix": confusion.tolist(),
    "per_class_metrics": per_class,
    "top_class_confusions": top_confusions,
    "class_balance": class_balance,
    "diagnosis": diagnosis,
}

with open(OUTPUT, "w") as f:
    json.dump(output, f, indent=2)

print(f"Diagnostics saved to {OUTPUT}")
print(f"Diagnosis: {diagnosis}")
print(f"Best threshold: {best_thresh} (F1={best_f1:.4f})")