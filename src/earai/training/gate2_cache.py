"""Gate 2 cache: exact browser DOM targets + CLIP embeddings."""
import hashlib
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from earai.training.browser_dataset import create_web_ui_dataset
from earai.training.gate2_teachers import create_gate2_teachers

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _cache_hash(config: dict) -> str:
    relevant = {
        "data_root": config.get("data_root"),
        "image_size": config.get("image_size"),
        "train_urls": config.get("train_urls", []),
        "val_urls": config.get("val_urls", []),
        "viewport_sizes": config.get("viewport_sizes", []),
        "scroll_fractions": config.get("scroll_fractions", []),
        "max_objects": config.get("max_objects", 24),
        # DOM filter logic version - bump when extract_dom_elements/filter_meaningful_elements changes
        "dom_filter_version": 2,
    }
    raw = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def build_teacher_cache(config: dict, device: str = "cpu",
                        cache_path: str = "artifacts/gate2_teacher_cache.pt") -> dict:
    path = Path(cache_path)
    expected_hash = _cache_hash(config)
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("config_hash") == expected_hash:
            return cache

    samples = create_web_ui_dataset(
        config, force_regenerate=bool(config.get("force_browser_dataset", False))
    )
    if not samples:
        raise RuntimeError("Gate 2 browser dataset is empty")

    teacher, _ = create_gate2_teachers(device)
    clip_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    targets = []
    clip_embeddings = []
    max_objects = int(config.get("max_objects", 24))

    with torch.no_grad():
        for sample in tqdm(samples, desc="Gate2 cache/CLIP"):
            image = Image.open(sample["image_path"]).convert("RGB")
            clip_img = clip_transform(image).unsqueeze(0).to(device)
            clip_emb = teacher.encode_clip(clip_img).squeeze(0).cpu()

            elems = []
            for elem in sample["ui_elements"][:max_objects]:
                elems.append({
                    "element_id": elem.get("element_id"),
                    "parent_id": elem.get("parent_id"),
                    "bbox": elem["bbox"],
                    "class_id": int(elem["class_id"]),
                    "text": elem.get("text", ""),
                    "style": elem.get("style", {}),
                })

            targets.append({
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "split": sample.get("split"),
                "domain": sample.get("domain", ""),
                "url": sample.get("url", ""),
                "viewport": sample.get("viewport"),
                "ui_elements": elems,
            })
            clip_embeddings.append(clip_emb)

    cache = {
        "config_hash": expected_hash,
        "targets": targets,
        "clip_embeddings": torch.stack(clip_embeddings),
        "num_images": len(targets),
        "train_count": sum(t.get("split") == "train" for t in targets),
        "val_count": sum(t.get("split") == "val" for t in targets),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


def _resize_with_padding(image: Image.Image, target_size: tuple) -> Image.Image:
    """Resize maintaining aspect ratio with center padding to target size."""
    target_h, target_w = target_size
    orig_w, orig_h = image.size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    padded = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    padded.paste(image, (left, top))
    
    return padded


def _style_vector(style: dict) -> list:
    vals = [
        *style.get("background", [0.5, 0.5, 0.5]),
        *style.get("foreground", [0.0, 0.0, 0.0]),
        style.get("radius", 0.0) / 50.0,
        (style.get("font_size", 16.0) - 12.0) / 48.0,
        (style.get("font_weight", 400.0) - 100.0) / 800.0,
        (style.get("line_height", 24.0) - 12.0) / 48.0,
    ]
    return [max(0.0, min(1.0, float(v))) for v in vals]


class CachedGate2Dataset(torch.utils.data.Dataset):
    def __init__(self, cache: dict, config: dict, split: str):
        self.config = config
        self.target_size = tuple(config.get("image_size", (224, 224)))
        self.max_objects = int(config.get("max_objects", 24))
        self.indices = [i for i, t in enumerate(cache["targets"]) if t.get("split") == split]
        if not self.indices:
            raise RuntimeError(f"Gate 2 cache has no {split} samples")
        self.targets = cache["targets"]
        self.clip_embeddings = cache["clip_embeddings"]

        self.student_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.raw_transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        target = self.targets[idx]
        image = Image.open(target["image_path"]).convert("RGB")
        elems = target["ui_elements"][:self.max_objects]

        # Resize with aspect ratio preservation + padding
        padded = _resize_with_padding(image, self.target_size)
        
        student_tensor = self.student_transform(padded)
        raw_tensor = self.raw_transform(padded)

        boxes = [e["bbox"] for e in elems]
        labels = [int(e["class_id"]) for e in elems]
        styles = [_style_vector(e.get("style", {})) for e in elems]

        return {
            "student": student_tensor,
            "raw": raw_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            "styles": torch.tensor(styles, dtype=torch.float32) if styles else torch.zeros(0, 10),
            "texts": [e.get("text", "") for e in elems],
            "element_ids": [e.get("element_id") for e in elems],
            "parent_ids": [e.get("parent_id") for e in elems],
            "image_id": target["image_id"],
            "domain": target.get("domain", ""),
            "url": target.get("url", ""),
            "clip_embedding": self.clip_embeddings[idx],
        }


def gate2_collate_fn_cached(batch):
    return {
        "student": torch.stack([b["student"] for b in batch]),
        "raw": torch.stack([b["raw"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "styles": [b["styles"] for b in batch],
        "texts": [b["texts"] for b in batch],
        "element_ids": [b["element_ids"] for b in batch],
        "parent_ids": [b["parent_ids"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "domains": [b["domain"] for b in batch],
        "urls": [b["url"] for b in batch],
        "clip_embeddings": torch.stack([b["clip_embedding"] for b in batch]),
    }


def create_cached_dataloader(config: dict, cache: dict, device: str = "cpu",
                             shuffle: bool = True, split: str = "train", drop_last: bool = None):
    dataset = CachedGate2Dataset(cache, config, split)
    if drop_last is None:
        drop_last = (split == "train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=False,
        collate_fn=gate2_collate_fn_cached,
        drop_last=drop_last,
    )