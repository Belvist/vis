#!/usr/bin/env python3
"""Gate 2: train and validate UI understanding on unseen domains."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from earai.training.browser_dataset import UI_CLASSES
from earai.training.gate2_cache import build_teacher_cache, create_cached_dataloader
from earai.training.gate2_loss import box_iou, create_gate2_loss
from earai.training.gate2_student import create_gate2_student


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _teacher_targets(batch, device):
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


def _decode_style(v):
    v = v.detach().cpu().float().numpy()
    return {
        "background": v[:3].tolist(),
        "foreground": v[3:6].tolist(),
        "radius": float(v[6] * 50.0),
        "font_size": float(v[7] * 48.0 + 12.0),
        "font_weight": float(v[8] * 800.0 + 100.0),
        "line_height": float(v[9] * 48.0 + 12.0),
    }


def _mean(values, missing=1e9):
    return float(np.mean(values)) if values else float(missing)


def compute_metrics(outputs, batches, objectness_threshold=0.5):
    ious_all = []
    tp = fp = fn = 0
    class_ok = class_n = 0
    bg_err, fg_err = [], []
    radius_err, font_err, weight_err, line_err = [], [], [], []
    parent_ok = parent_n = 0

    for out, batch in zip(outputs, batches):
        pred_boxes_all = out["bboxes_xyxy"]
        pred_labels_all = out["class_logits"].argmax(-1)
        pred_obj_all = out["objectness"]
        pred_style_all = out["style"]
        pred_hier_all = out["hierarchy"]

        for b in range(pred_boxes_all.shape[0]):
            active = torch.nonzero(pred_obj_all[b] >= objectness_threshold, as_tuple=False).flatten()
            gt_boxes = batch["boxes"][b].to(pred_boxes_all.device)
            gt_labels = batch["labels"][b].to(pred_boxes_all.device)
            gt_styles = batch["styles"][b].to(pred_boxes_all.device)
            gt_element_ids = batch["element_ids"][b]
            gt_parent_ids = batch["parent_ids"][b]

            if len(gt_boxes) == 0:
                fp += len(active)
                continue
            if len(active) == 0:
                fn += len(gt_boxes)
                continue

            pred_boxes = pred_boxes_all[b, active]
            pred_labels = pred_labels_all[b, active]
            pred_styles = pred_style_all[b, active]
            ious = box_iou(pred_boxes, gt_boxes)
            pi, ti = linear_sum_assignment((-ious).detach().cpu().numpy())

            assigned_pred = set()
            assigned_gt = set()
            good_target_to_query = {}

            for p, t in zip(pi, ti):
                assigned_pred.add(int(p))
                assigned_gt.add(int(t))
                iou = float(ious[p, t].item())
                if iou < 0.5:
                    fp += 1
                    fn += 1
                    continue

                tp += 1
                ious_all.append(iou)
                query_idx = int(active[p].item())
                good_target_to_query[int(t)] = query_idx

                class_n += 1
                if int(pred_labels[p].item()) == int(gt_labels[t].item()):
                    class_ok += 1

                ps = pred_styles[p]
                ts = gt_styles[t]
                bg_err.append(float(torch.mean(torch.abs(ps[:3] - ts[:3])).item()))
                fg_err.append(float(torch.mean(torch.abs(ps[3:6] - ts[3:6])).item()))
                radius_err.append(float(torch.abs(ps[6] - ts[6]).item() * 50.0))
                font_err.append(float(torch.abs(ps[7] - ts[7]).item() * 48.0))
                weight_err.append(float(torch.abs(ps[8] - ts[8]).item() * 800.0))
                line_err.append(float(torch.abs(ps[9] - ts[9]).item() * 48.0))

            fp += len(active) - len(assigned_pred)
            fn += len(gt_boxes) - len(assigned_gt)

            id_to_target = {
                eid: i for i, eid in enumerate(gt_element_ids) if eid is not None
            }
            candidate_queries = list(good_target_to_query.values())
            if candidate_queries:
                candidate_tensor = torch.tensor(
                    candidate_queries, device=pred_hier_all.device, dtype=torch.long
                )
                for child_t, child_q in good_target_to_query.items():
                    parent_id = gt_parent_ids[child_t]
                    parent_t = id_to_target.get(parent_id)
                    if parent_t is None or parent_t not in good_target_to_query:
                        continue
                    expected_q = good_target_to_query[parent_t]
                    scores = pred_hier_all[b, candidate_tensor, child_q]
                    best_pos = int(torch.argmax(scores).item())
                    predicted_q = candidate_queries[best_pos]
                    parent_n += 1
                    if float(scores[best_pos].item()) >= 0.5 and predicted_q == expected_q:
                        parent_ok += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "mean_iou": _mean(ious_all, 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_accuracy": class_ok / max(class_n, 1),
        "bg_color_mae": _mean(bg_err),
        "fg_color_mae": _mean(fg_err),
        "radius_mae_px": _mean(radius_err),
        "font_size_mae_px": _mean(font_err),
        "font_weight_mae": _mean(weight_err),
        "line_height_mae_px": _mean(line_err),
        "parent_relation_accuracy": parent_ok / max(parent_n, 1),
        "parent_relations_evaluated": parent_n,
        "tp": tp, "fp": fp, "fn": fn,
        "objectness_threshold": float(objectness_threshold),
    }


def passes_gate(metrics, config):
    return (
        metrics["precision"] >= float(config.get("pass_precision", 0.85)) and
        metrics["recall"] >= float(config.get("pass_recall", 0.85)) and
        metrics["class_accuracy"] >= float(config.get("pass_class_accuracy", 0.90)) and
        metrics["mean_iou"] >= float(config.get("pass_mean_iou", 0.70)) and
        metrics["radius_mae_px"] <= float(config.get("pass_radius_mae_px", 4.0)) and
        metrics["font_size_mae_px"] <= float(config.get("pass_font_size_mae_px", 3.0)) and
        metrics["parent_relation_accuracy"] >= float(config.get("pass_parent_accuracy", 0.85))
    )


class Gate2Trainer:
    def __init__(self, config, device, cache):
        self.config = config
        self.device = torch.device(device)
        self.student = create_gate2_student(config).to(self.device)
        self.loss_fn = create_gate2_loss(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(config.get("lr", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        self.train_loader = create_cached_dataloader(
            config, cache, device, shuffle=True, split="train"
        )
        self.val_loader = create_cached_dataloader(
            config, cache, device, shuffle=False, split="val"
        )
        self.initial_loss = None
        self.history = []
        self.best_state = None
        self.best_f1 = -1.0

    def train_epoch(self):
        self.student.train()
        totals = {}
        for batch in tqdm(self.train_loader, desc="Gate2 train"):
            images = batch["student"].to(self.device)
            out = self.student(images)
            losses = self.loss_fn(out["tokens"], out, _teacher_targets(batch, self.device))
            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            for key, value in losses.items():
                if key == "num_matched":
                    continue
                totals.setdefault(key, []).append(float(value.detach().cpu().item()))

        avg = {k: float(np.mean(v)) for k, v in totals.items()}
        if self.initial_loss is None:
            self.initial_loss = avg["total"]
        self.history.append(avg)
        return avg

    @torch.no_grad()
    def evaluate(self, loader, save_predictions=False):
        self.student.eval()
        outputs = []
        batches = []
        examples = []

        for batch in loader:
            out = self.student(batch["student"].to(self.device))
            cpu_out = {
                key: value.detach().cpu()
                for key, value in out.items()
                if torch.is_tensor(value)
            }
            outputs.append(cpu_out)
            batches.append(batch)

            if save_predictions and len(examples) < 20:
                for b in range(len(batch["image_ids"])):
                    if len(examples) >= 20:
                        break
                    pred_mask = cpu_out["objectness"][b] >= 0.5
                    preds = []
                    for q in torch.nonzero(pred_mask, as_tuple=False).flatten().tolist():
                        cls = int(cpu_out["class_logits"][b, q].argmax().item())
                        if cls >= len(UI_CLASSES):
                            continue
                        preds.append({
                            "class": UI_CLASSES[cls],
                            "bbox": cpu_out["bboxes_xyxy"][b, q].tolist(),
                            "objectness": float(cpu_out["objectness"][b, q].item()),
                            "style": _decode_style(cpu_out["style"][b, q]),
                        })
                    truth = []
                    for j in range(len(batch["boxes"][b])):
                        cls = int(batch["labels"][b][j].item())
                        truth.append({
                            "element_id": batch["element_ids"][b][j],
                            "parent_id": batch["parent_ids"][b][j],
                            "class": UI_CLASSES[cls],
                            "bbox": batch["boxes"][b][j].tolist(),
                            "text": batch["texts"][b][j],
                            "style_normalized": batch["styles"][b][j].tolist(),
                        })
                    examples.append({
                        "image_id": batch["image_ids"][b],
                        "domain": batch["domains"][b],
                        "url": batch["urls"][b],
                        "truth": truth,
                        "earai": preds,
                    })

        metrics = compute_metrics(outputs, batches)
        return (metrics, examples) if save_predictions else metrics


def main(config, requested_device="auto"):
    device = _device(requested_device)
    cache = build_teacher_cache(config, device="cpu")

    min_train = int(config.get("min_train_samples", 0))
    min_val = int(config.get("min_val_samples", 0))
    if cache.get("train_count", 0) < min_train or cache.get("val_count", 0) < min_val:
        raise RuntimeError(
            f"Gate2 dataset too small: train={cache.get('train_count', 0)} "
            f"val={cache.get('val_count', 0)}; required train>={min_train}, val>={min_val}"
        )

    trainer = Gate2Trainer(config, device, cache)
    epochs = int(config.get("epochs", 50))
    eval_every = int(config.get("eval_every", 5))
    epochs_ran = 0

    for epoch in range(epochs):
        train_loss = trainer.train_epoch()
        epochs_ran = epoch + 1
        print(f"Epoch {epoch}: {train_loss}")

        if epoch % eval_every == 0 or epoch == epochs - 1:
            val = trainer.evaluate(trainer.val_loader)
            print(f"VAL {epoch}: {val}")
            if val["f1"] > trainer.best_f1:
                trainer.best_f1 = val["f1"]
                trainer.best_state = copy.deepcopy(trainer.student.state_dict())
            if passes_gate(val, config):
                print("Gate 2 validation criteria reached; stopping early.")
                break

    if trainer.best_state is not None:
        trainer.student.load_state_dict(trainer.best_state)

    final_val, examples = trainer.evaluate(trainer.val_loader, save_predictions=True)
    final_train = trainer.evaluate(trainer.train_loader)
    passed = passes_gate(final_val, config)

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    report = {
        "device": device,
        "epochs_ran": epochs_ran,
        "train_samples": len(trainer.train_loader.dataset),
        "val_samples": len(trainer.val_loader.dataset),
        "initial_loss": trainer.initial_loss,
        "final_train_loss": trainer.history[-1]["total"],
        "train_metrics": final_train,
        "val_metrics": final_val,
        "PASS": passed,
    }
    with open(artifacts / "gate2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(artifacts / "gate2_predictions.json", "w") as f:
        json.dump(examples, f, indent=2)
    torch.save(
        {"student_state": trainer.student.state_dict(), "config": config, "report": report},
        artifacts / "gate2_best.pt",
    )

    print(json.dumps(report, indent=2))
    print("PASS" if passed else "FAIL")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate2.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--regenerate-dataset", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.regenerate_dataset:
        cfg["force_browser_dataset"] = True
    main(cfg, args.device)
