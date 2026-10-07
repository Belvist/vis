"""Gate 2 Teacher Cache - precompute browser DOM ground truth + CLIP embeddings once"""
import torch
import torch.nn as nn
from typing import Dict, List
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from earai.training.gate2_teachers import create_gate2_teachers
from earai.training.browser_dataset import create_web_ui_dataset, UI_CLASS_TO_IDX

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_teacher_cache(config: dict, device: str = 'cpu', cache_path: str = 'artifacts/gate2_teacher_cache.pt') -> Dict:
    """
    Precompute CLIP embeddings and store browser DOM ground truth.
    NO teacher inference for bbox/class/style - those come from DOM.
    """
    cache_path = Path(cache_path)

    # Check if cache exists and is valid
    if Path(cache_path).exists():
        print(f"Loading teacher cache from {cache_path}...")
        cache = torch.load(cache_path, map_location=device, weights_only=False)
        if cache.get('config_hash') == _config_hash(config):
            print(f"Cache valid: {len(cache['targets'])} images")
            return cache
        print("Config changed, rebuilding cache...")

    print("Building teacher cache from browser dataset...")

    # Load browser dataset
    samples = create_web_ui_dataset(config, force_regenerate=config.get('force_browser_dataset', False))

    if not samples:
        raise RuntimeError("Browser dataset empty. Cannot build cache.")

    # Create CLIP teacher for embeddings only
    teachers, _ = create_gate2_teachers(device)

    # Transforms for CLIP
    clip_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    # Precompute CLIP embeddings only
    all_targets = []
    all_image_ids = []
    all_clip_embeddings = []

    with torch.no_grad():
        for sample in tqdm(samples, desc="CLIP encoding"):
            img_id = sample['image_id']
            image_path = sample['image_path']

            # Load image for CLIP
            image = Image.open(image_path).convert('RGB')
            clip_img = clip_transform(image).unsqueeze(0).to(device)

            # CLIP embedding
            clip_emb = teachers.encode_clip(clip_img).squeeze(0).cpu()

            # Build target from DOM ground truth (NOT from teacher inference)
            ui_elements = []
            for elem in sample['ui_elements']:
                ui_elements.append({
                    'bbox': elem['bbox'],          # already normalized [0,1]
                    'class_id': elem['class_id'],  # UI class index 0-14
                    'text': elem.get('text', ''),
                    'parent_id': elem.get('parent_id'),
                    'style': {
                        'background': elem['style'].get('background', [0.5, 0.5, 0.5]),
                        'foreground': elem['style'].get('foreground', [0, 0, 0]),
                        'radius': elem['style'].get('radius', 0),
                        'font_size': elem['style'].get('font_size', 16),
                        'font_weight': elem['style'].get('font_weight', 400),
                        'line_height': elem['style'].get('line_height', 24),
                    }
                })

            target = {
                'image_id': img_id,
                'image_path': image_path,
                'viewport': sample['viewport'],
                'ui_elements': ui_elements,
                'containers': [],
                'clip_embedding': clip_emb.numpy()
            }

            all_targets.append(target)
            all_image_ids.append(img_id)
            all_clip_embeddings.append(clip_emb)

    cache = {
        'config_hash': _config_hash(config),
        'targets': all_targets,
        'image_ids': all_image_ids,
        'clip_embeddings': torch.stack(all_clip_embeddings),
        'num_images': len(all_targets),
    }

    Path(cache_path).parent.mkdir(exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Teacher cache saved: {cache_path} ({len(all_targets)} images)")

    return cache


def _config_hash(config: dict) -> str:
    """Simple hash of relevant config for cache validation"""
    import hashlib
    key = f"{config.get('data_root')}{config.get('image_size')}{config.get('split')}{config.get('force_browser_dataset')}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


class CachedGate2Dataset(torch.utils.data.Dataset):
    """Dataset that returns student inputs + precomputed DOM ground truth + CLIP embeddings."""

    def __init__(self, cache: Dict, config: dict, device: str = 'cpu'):
        self.cache = cache
        self.config = config
        self.device = device
        self.targets = cache['targets']
        self.image_ids = cache['image_ids']
        self.clip_embeddings = cache['clip_embeddings']

        self.image_size = tuple(config.get('image_size', (224, 224)))
        self.max_objects = config.get('max_objects', 50)

        # Transforms
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        self.raw_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
        ])

        self.clip_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD)
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict:
        target = self.targets[idx]
        img_id = self.image_ids[idx]

        # Load real image from browser-rendered path
        image_path = target['image_path']
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        student_img = self.transform(image)
        raw_img = self.raw_transform(image)
        clip_img = self.clip_preprocess(image)

        # Prepare UI elements from DOM ground truth
        boxes = []
        labels = []
        styles = []
        texts = []
        parent_ids = []

        for elem in target['ui_elements'][:self.max_objects]:
            bbox = elem['bbox']
            boxes.append(bbox)
            labels.append(elem['class_id'])
            texts.append(elem.get('text', ''))
            parent_ids.append(elem.get('parent_id'))

            style = elem['style']
            style_vec = [
                *style['background'],      # 3
                *style['foreground'],      # 3
                style['radius'] / 50.0,    # 1
                (style['font_size'] - 12) / 48.0,    # 1
                (style['font_weight'] - 100) / 800.0,  # 1
                (style['line_height'] - 12) / 48.0,   # 1
            ]
            styles.append(style_vec)

        return {
            'student': student_img,
            'raw': raw_img,
            'clip': clip_img,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'styles': torch.tensor(styles, dtype=torch.float32) if styles else torch.zeros(0, 10),
            'texts': texts,
            'parent_ids': parent_ids,
            'image_id': img_id,
            'clip_embedding': self.clip_embeddings[idx]
        }


def create_cached_dataloader(config: dict, cache: Dict, device: str = 'cpu', shuffle: bool = True):
    """Create dataloader with cached DOM ground truth + CLIP embeddings"""
    dataset = CachedGate2Dataset(cache, config, device)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        pin_memory=False,
        collate_fn=gate2_collate_fn_cached,
        drop_last=True
    )


def gate2_collate_fn_cached(batch):
    """Collate function for cached dataset"""
    return {
        'student': torch.stack([b['student'] for b in batch]),
        'raw': torch.stack([b['raw'] for b in batch]),
        'clip': torch.stack([b['clip'] for b in batch]),
        'boxes': [b['boxes'] for b in batch],
        'labels': [b['labels'] for b in batch],
        'styles': [b['styles'] for b in batch],
        'texts': [b['texts'] for b in batch],
        'parent_ids': [b['parent_ids'] for b in batch],
        'image_ids': [b['image_id'] for b in batch],
        'clip_embeddings': torch.stack([b['clip_embedding'] for b in batch]),
    }