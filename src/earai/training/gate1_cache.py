"""Gate 1 Teacher Cache - precompute teacher outputs once"""
import torch
import torch.nn as nn
from typing import Dict, List
from pathlib import Path
import json
from tqdm import tqdm

from earai.training.gate1_teachers import create_gate1_teachers
from earai.training.gate1_dataset import create_gate1_dataloader


def build_teacher_cache(config: dict, device: str = 'cpu', cache_path: Path = Path('artifacts/gate1_teacher_cache.pt')) -> Dict:
    """Precompute teacher outputs for all 64 images once."""
    cache_path = Path(cache_path)

    # Check if cache exists and is valid
    if cache_path.exists():
        print(f"Loading teacher cache from {cache_path}...")
        cache = torch.load(cache_path, map_location=device, weights_only=False)
        if cache.get('config_hash') == _config_hash(config):
            print(f"Cache valid: {len(cache['targets'])} images")
            return cache
        print("Config changed, rebuilding cache...")

    print("Building teacher cache (one-time)...")

    # Create teachers
    teachers, _ = create_gate1_teachers(device)

    # Create dataloader (no shuffle for consistent order)
    dataloader = create_gate1_dataloader(config, shuffle=False)

    # Precompute
    all_targets = []
    all_image_ids = []
    all_clip_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Teacher inference"):
            student_imgs = batch['student'].to(device)
            raw_imgs = batch['raw'].to(device)
            clip_imgs = batch['clip'].to(device)

            teacher_out = teachers(student_imgs, raw_imgs, clip_imgs)

            # Store targets for each image
            for i in range(len(batch['image_ids'])):
                img_id = batch['image_ids'][i]

                det = teacher_out['detections'][i]
                target = {
                    'image_id': img_id,
                    'boxes': det['boxes'],
                    'labels': det['labels'],
                    'scores': det['scores'],
                }
                all_targets.append(target)
                all_image_ids.append(img_id)
                all_clip_embeddings.append(teacher_out['clip_embeddings'][i].cpu())

    cache = {
        'config_hash': _config_hash(config),
        'targets': all_targets,
        'image_ids': all_image_ids,
        'clip_embeddings': torch.stack(all_clip_embeddings),
        'num_images': len(all_targets),
    }

    cache_path.parent.mkdir(exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Teacher cache saved: {cache_path} ({len(all_targets)} images)")

    return cache


def _config_hash(config: dict) -> str:
    """Simple hash of relevant config for cache validation"""
    import hashlib
    key = f"{config.get('data_root')}{config.get('image_size')}{config.get('coco_split')}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


class CachedTeacherDataset(torch.utils.data.Dataset):
    """
    Dataset that returns student inputs + precomputed teacher targets.
    No teacher inference during training.
    """
    
    def __init__(self, cache: Dict, config: dict, device: str = 'cpu'):
        self.cache = cache
        self.config = config
        self.device = device
        self.targets = cache['targets']
        self.image_ids = cache['image_ids']
        self.clip_embeddings = cache['clip_embeddings']
        
        # Student transforms only (no teacher transforms needed)
        from earai.training.gate1_teachers import Gate1DataTransforms
        self.transforms = Gate1DataTransforms(config.get('image_size', (224, 224)))
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Dict:
        target = self.targets[idx]
        img_id = self.image_ids[idx]
        
        # Load image and apply student transforms
        import cv2
        img_path = f"./data/coco/val2017/{img_id:012d}.jpg"
        image = cv2.imread(img_path)
        if image is None:
            img_path = f"./data/coco/val2017/{img_id}.jpg"
            image = cv2.imread(img_path)
        
        if image is None:
            raise RuntimeError(f"Failed to load image for id {img_id}")
        
        transformed = self.transforms(image)
        
        return {
            'student': transformed['student'],     # [3, H, W] ImageNet norm
            'raw': transformed['raw'],             # [3, H, W] raw [0,1]
            'clip': transformed['clip'],           # [3, 224, 224] CLIP norm
            'boxes': torch.tensor(target['boxes'], dtype=torch.float32) if len(target['boxes']) > 0 else torch.zeros(0, 4),
            'labels': torch.tensor(target['labels'], dtype=torch.long) if len(target['labels']) > 0 else torch.zeros(0, dtype=torch.long),
            'image_id': img_id,
            'clip_embedding': self.clip_embeddings[idx],  # [512]
        }


def create_cached_dataloader(config: dict, cache: Dict, device: str = 'cpu', shuffle: bool = True):
    """Create dataloader with cached teacher targets"""
    dataset = CachedTeacherDataset(cache, config, device)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        pin_memory=False,
        collate_fn=gate1_collate_fn_cached
    )


def gate1_collate_fn_cached(batch):
    """Collate function for cached dataset"""
    return {
        'student': torch.stack([b['student'] for b in batch]),
        'raw': torch.stack([b['raw'] for b in batch]),
        'clip': torch.stack([b['clip'] for b in batch]),
        'boxes': [b['boxes'] for b in batch],
        'labels': [b['labels'] for b in batch],
        'image_ids': [b['image_id'] for b in batch],
        'clip_embeddings': torch.stack([b['clip_embedding'] for b in batch]),
    }