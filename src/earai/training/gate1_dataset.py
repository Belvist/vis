"""Gate 1 Fixed COCO Dataset - 64 images for overfit proof"""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from earai.training.gate1_teachers import Gate1DataTransforms


# Fixed 64 COCO image IDs for reproducible overfit test
# These are VALID COCO 2017 val set image IDs (verified to exist)
GATE1_IMAGE_IDS = [
    397133, 369370, 344059, 340930, 453708, 205834, 286553, 127955,
    111951, 430056, 311909, 399205, 173830, 15338, 95155, 507797,
    320642, 427160, 211069, 439525, 158956, 192904, 411530, 27982,
    327592, 523194, 167353, 348488, 500477, 449579, 491757, 191845,
    312489, 198915, 125778, 473406, 408112, 491130, 395575, 441491,
    235241, 245173, 574823, 333069, 150930, 250127, 226984, 529762,
    550084, 17899, 121744, 309964, 279278, 217872, 202339, 57760,
    515982, 446117, 511453, 193181, 306893, 539883, 522393, 442993
]

# COCO category names (canonical 0-79)
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


class Gate1COCODataset(Dataset):
    """
    Fixed 64-image COCO dataset for Gate 1 overfit proof.
    No synthetic fallback - raises if COCO not available.
    """
    
    def __init__(self, 
                 coco_root: str,
                 split: str = 'val2017',
                 image_size: Tuple[int, int] = (224, 224),
                 max_objects: int = 16):
        
        self.coco_root = Path(coco_root)
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        
        # Load COCO annotations
        self.coco = self._load_coco()
        self.transforms = Gate1DataTransforms(image_size)
        
        # Filter to our fixed 64 images
        self.samples = self._filter_samples()
        
        if len(self.samples) != 64:
            raise RuntimeError(f"Expected 64 Gate 1 images, got {len(self.samples)}. Check COCO data.")
    
    def _load_coco(self):
        """Load COCO annotations"""
        ann_file = self.coco_root / 'annotations' / f'instances_{self.split}.json'
        if not ann_file.exists():
            raise RuntimeError(f"COCO annotations not found: {ann_file}")
        
        from pycocotools.coco import COCO
        return COCO(str(ann_file))
    
    def _filter_samples(self) -> List[Dict]:
        """Filter COCO to our fixed 64 images"""
        samples = []
        
        for img_id in GATE1_IMAGE_IDS:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Filter valid annotations
            valid_anns = []
            for ann in anns:
                if ann.get('iscrowd', 0) == 0 and ann['area'] > 100:
                    x, y, w, h = ann['bbox']
                    if w > 10 and h > 10:
                        valid_anns.append(ann)
            
            # Limit objects
            valid_anns = valid_anns[:self.max_objects]
            
            img_path = self.coco_root.parent / 'val2017' / img_info['file_name']
            if img_path.exists():
                samples.append({
                    'image_id': img_id,
                    'image_path': str(img_path),
                    'annotations': valid_anns,
                    'width': img_info['width'],
                    'height': img_info['height']
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise RuntimeError(f"Failed to load image: {sample['image_path']}")
        
        # Apply transforms (returns student, raw, clip tensors)
        transformed = self.transforms(image)
        
        # Process annotations to normalized xyxy + canonical labels
        boxes = []
        labels = []
        for ann in sample['annotations']:
            x, y, w, h = ann['bbox']
            # Normalize to [0,1]
            boxes.append([
                x / sample['width'],
                y / sample['height'],
                (x + w) / sample['width'],
                (y + h) / sample['height']
            ])
            # COCO category ID -> canonical 0-79
            cat_id = ann['category_id']
            # COCO mapping
            coco_mapping = {
                1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
                21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
                33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
                42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
                52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
                61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
                74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
                84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
            }
            labels.append(coco_mapping.get(cat_id, 0))
        
        return {
            'student': transformed['student'],     # [3, H, W] ImageNet norm
            'raw': transformed['raw'],             # [3, H, W] raw [0,1]
            'clip': transformed['clip'],           # [3, 224, 224] CLIP norm
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': sample['image_id'],
            'width': sample['width'],
            'height': sample['height']
        }


def gate1_collate_fn(batch):
    """Collate function for Gate 1 dataset"""
    return {
        'student': torch.stack([b['student'] for b in batch]),
        'raw': torch.stack([b['raw'] for b in batch]),
        'clip': torch.stack([b['clip'] for b in batch]),
        'boxes': [b['boxes'] for b in batch],  # List of [N_i, 4]
        'labels': [b['labels'] for b in batch],  # List of [N_i]
        'image_ids': [b['image_id'] for b in batch],
        'widths': [b['width'] for b in batch],
        'heights': [b['height'] for b in batch]
    }


def create_gate1_dataloader(config: dict, shuffle: bool = True) -> DataLoader:
    """Create Gate 1 dataloader"""
    dataset = Gate1COCODataset(
        coco_root=config['data_root'],
        split=config.get('coco_split', 'val2017'),
        image_size=config.get('image_size', (224, 224)),
        max_objects=config.get('max_objects', 16)
    )
    
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        pin_memory=False,  # MPS doesn't support
        collate_fn=gate1_collate_fn
    )