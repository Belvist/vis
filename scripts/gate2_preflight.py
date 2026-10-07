#!/usr/bin/env python3
"""Fast Gate 2 preflight: dataset/split/model/loss/backward contract."""
import argparse
import json

import torch
import yaml

from earai.training.gate2_cache import build_teacher_cache, create_cached_dataloader
from earai.training.gate2_loss import create_gate2_loss
from earai.training.gate2_student import create_gate2_student


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def teacher_targets(batch, device):
    targets = []
    for i in range(len(batch["boxes"])):
        elems = []
        for j in range(len(batch["boxes"][i])):
            elems.append({
                "bbox": batch["boxes"][i][j].to(device),
                "class_id": batch["labels"][i][j].to(device),
                "style": batch["styles"][i][j].to(device),
                "element_id": batch["element_ids"][i][j],
                "parent_id": batch["parent_ids"][i][j],
            })
        targets.append({"ui_elements": elems})
    return {
        "targets": targets,
        "clip_embeddings": batch["clip_embeddings"].to(device),
    }


def main(config_path: str, regenerate: bool):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if regenerate:
        cfg["force_browser_dataset"] = True

    device = auto_device()
    cache = build_teacher_cache(cfg, device="cpu")

    train_domains = {t.get("domain") for t in cache["targets"] if t.get("split") == "train"}
    val_domains = {t.get("domain") for t in cache["targets"] if t.get("split") == "val"}
    overlap = sorted(train_domains & val_domains)
    if overlap:
        raise RuntimeError(f"Domain leakage: {overlap}")

    train_loader = create_cached_dataloader(cfg, cache, device, shuffle=False, split="train")
    val_loader = create_cached_dataloader(cfg, cache, device, shuffle=False, split="val")

    model = create_gate2_student(cfg).to(device)
    loss_fn = create_gate2_loss(cfg).to(device)

    batch = next(iter(train_loader))
    out = model(batch["student"].to(device))
    expected_tokens = int(cfg.get("num_scene_tokens", 32))
    if out["tokens"].shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} tokens, got {out['tokens'].shape[1]}"
        )
    if out["style"].shape[-1] != 10:
        raise RuntimeError(f"Expected style_dim=10, got {out['style'].shape[-1]}")

    losses = loss_fn(out["tokens"], out, teacher_targets(batch, device))
    if not torch.isfinite(losses["total"]):
        raise RuntimeError(f"Non-finite loss: {losses}")

    model.zero_grad(set_to_none=True)
    losses["total"].backward()
    grad_params = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    if grad_params == 0:
        raise RuntimeError("No gradients reached the Gate 2 model")

    result = {
        "device": device,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "train_domains": sorted(train_domains),
        "val_domains": sorted(val_domains),
        "domain_overlap": overlap,
        "tokens": list(out["tokens"].shape),
        "boxes": list(out["bboxes_xyxy"].shape),
        "style": list(out["style"].shape),
        "hierarchy": list(out["hierarchy"].shape),
        "loss_total": float(losses["total"].detach().cpu().item()),
        "loss_type": float(losses["type"].detach().cpu().item()),
        "loss_bbox": float(losses["bbox"].detach().cpu().item()),
        "loss_giou": float(losses["giou"].detach().cpu().item()),
        "loss_style": float(losses["style"].detach().cpu().item()),
        "loss_hierarchy": float(losses["hierarchy"].detach().cpu().item()),
        "grad_params": grad_params,
        "PASS": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate2.yaml")
    parser.add_argument("--regenerate-dataset", action="store_true")
    args = parser.parse_args()
    main(args.config, args.regenerate_dataset)
