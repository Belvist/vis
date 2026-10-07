"""Gate 2 browser-rendered Web UI dataset."""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from earai.training.browser_dataset import create_web_ui_dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


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


class Gate2BrowserDataset(Dataset):
    def __init__(self, config: dict, split: str = "train"):
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split: {split}")
        self.config = config
        self.split = split
        self.image_size = tuple(config.get("image_size", (224, 224)))
        self.max_objects = int(config.get("max_objects", 32))

        self.student_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.raw_transform = T.Compose([T.Resize(self.image_size), T.ToTensor()])
        self.clip_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

        samples = create_web_ui_dataset(
            config, force_regenerate=bool(config.get("force_browser_dataset", False))
        )
        self.samples = [s for s in samples if s.get("split") == split]
        if not self.samples:
            raise RuntimeError(f"Gate 2 {split} dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        elems = sample["ui_elements"][:self.max_objects]
        boxes = [e["bbox"] for e in elems]
        labels = [int(e["class_id"]) for e in elems]
        styles = [_style_vector(e.get("style", {})) for e in elems]

        return {
            "student": self.student_transform(image),
            "raw": self.raw_transform(image),
            "clip": self.clip_transform(image),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            "styles": torch.tensor(styles, dtype=torch.float32) if styles else torch.zeros(0, 10),
            "texts": [e.get("text", "") for e in elems],
            "element_ids": [e.get("element_id") for e in elems],
            "parent_ids": [e.get("parent_id") for e in elems],
            "image_id": sample["image_id"],
            "domain": sample.get("domain", ""),
            "url": sample.get("url", ""),
            "source": "browser",
        }


def gate2_collate_fn(batch):
    return {
        "student": torch.stack([b["student"] for b in batch]),
        "raw": torch.stack([b["raw"] for b in batch]),
        "clip": torch.stack([b["clip"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "styles": [b["styles"] for b in batch],
        "texts": [b["texts"] for b in batch],
        "element_ids": [b["element_ids"] for b in batch],
        "parent_ids": [b["parent_ids"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "domains": [b["domain"] for b in batch],
        "urls": [b["url"] for b in batch],
    }


def create_gate2_dataloader(config: dict, shuffle: bool = True, split: str = "train") -> DataLoader:
    dataset = Gate2BrowserDataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=gate2_collate_fn,
        pin_memory=False,
        drop_last=(split == "train"),
    )
