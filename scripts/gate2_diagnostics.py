#!/usr/bin/env python3
"""Truthful Gate 2 diagnostics on a checkpoint and held-out domain split.

Uses the same IoU, matching, and objectness implementation as training.
The threshold sweep is diagnostic only; acceptance always uses 0.50.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earai.training.gate2_cache import build_teacher_cache, create_cached_dataloader
from earai.training.gate2_student import create_gate2_student
from gate2_train import compute_metrics, passes_gate


def _device(name):
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _collect(model, loader, device):
    outputs, batches = [], []
    class_counts = Counter()
    for batch in loader:
        out = model(batch["student"].to(device))
        outputs.append({
            k: v.detach().cpu() for k, v in out.items() if torch.is_tensor(v)
        })
        batches.append(batch)
        for labels in batch["labels"]:
            class_counts.update(int(n) for n in labels.tolist())
    return outputs, batches, class_counts


def main(config_path="configs/gate2.yaml", checkpoint="artifacts/gate2_best.pt",
         output="artifacts/gate2_diagnostics.json", requested_device="auto"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Checkpoint files are trusted local artifacts; do not open untrusted .pt files.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trained_config = ckpt.get("config")
    if not isinstance(trained_config, dict):
        raise RuntimeError("Checkpoint has no training configuration; provenance unknown")
    contracts = (
        "num_scene_tokens", "max_objects", "image_size", "num_ui_classes",
        "style_dim", "feature_dim", "backbone_pretrained",
    )
    changes = {
        key: {"checkpoint": trained_config.get(key), "current": config.get(key)}
        for key in contracts if trained_config.get(key) != config.get(key)
    }
    if changes:
        raise RuntimeError(
            "Checkpoint architecture/data contract differs from current config. "
            "Do not present metrics for incompatible checkpoints; retrain first. "
            + json.dumps(changes)
        )

    device = _device(requested_device)
    model = create_gate2_student(config).to(device)
    model.load_state_dict(ckpt["student_state"], strict=True)
    model.eval()

    cache = build_teacher_cache(config, device="cpu")
    train_loader = create_cached_dataloader(
        config, cache, device=device, split="train", shuffle=False, drop_last=False
    )
    val_loader = create_cached_dataloader(
        config, cache, device=device, split="val", shuffle=False, drop_last=False
    )
    train_out, train_batches, train_counts = _collect(model, train_loader, device)
    val_out, val_batches, val_counts = _collect(model, val_loader, device)
    assert not (
        {d for b in train_batches for d in b["domains"]} &
        {d for b in val_batches for d in b["domains"]}
    ), "Train/validation domain leakage"

    fixed = compute_metrics(val_out, val_batches, objectness_threshold=0.50)
    fixed_train = compute_metrics(train_out, train_batches, objectness_threshold=0.50)
    thresholds = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    sweep = {
        str(t): compute_metrics(val_out, val_batches, objectness_threshold=t)
        for t in thresholds
    }
    best = max(sweep, key=lambda t: sweep[t]["f1"])

    report = {
        "checkpoint": str(checkpoint),
        "device": device,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "fixed_threshold": 0.50,
        "train_metrics": fixed_train,
        "val_metrics": fixed,
        "PASS_at_fixed_threshold": passes_gate(fixed, config),
        "val_threshold_sweep_diagnostic_only": sweep,
        "best_val_threshold_not_for_acceptance": float(best),
        "train_class_counts": dict(sorted(train_counts.items())),
        "val_class_counts": dict(sorted(val_counts.items())),
        "note": "Threshold selected on validation is not a held-out test result. "
                "Use fixed 0.50 for acceptance, and evaluate genuine unseen images.",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate2.yaml")
    parser.add_argument("--checkpoint", default="artifacts/gate2_best.pt")
    parser.add_argument("--output", default="artifacts/gate2_diagnostics.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.device)
