"""Gate 2 Dataset - Browser-rendered Web UI screenshots with DOM ground truth"""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple
import random
from PIL import Image

from earai.training.browser_dataset import create_web_ui_dataset, UI_CLASS_TO_IDX


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Gate2BrowserDataset(Dataset):
    """Dataset using browser-rendered web UI screenshots with DOM ground truth"""
    
    def __init__(self, config: dict, split: str = 'train'):
        self.config = config
        self.split = split
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
        
        # CLIP preprocessing
        self.clip_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                       [0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Load browser dataset
        print(f"Loading browser-rendered web UI dataset from {config.get('data_root')}...")
        self.samples = create_web_ui_dataset(config, force_regenerate=config.get('force_browser_dataset', False))
        
        if not self.samples:
            raise RuntimeError(
                "Browser dataset generation failed or returned empty. "
                "Check URLs and network connectivity. "
                "No synthetic fallback allowed for Gate 2."
            )
        
        # Split train/val
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.9)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        student_img = self.transform(image)
        raw_img = self.raw_transform(image)
        clip_img = self.clip_preprocess(image)
        
        # Process UI elements
        boxes = []
        labels = []
        styles = []
        texts = []
        
        for elem in sample['ui_elements'][:self.max_objects]:
            bbox = elem['bbox']
            # bbox is already normalized [0,1]
            boxes.append(bbox)
            labels.append(elem['class_id'])
            texts.append(elem.get('text', ''))
            
            style = elem.get('style', {})
            style_vec = [
                *style.get('background', [0.5, 0.5, 0.5]),
                *style.get('foreground', [0, 0, 0]),
                style.get('radius', 0) / 50.0,
                (style.get('font_size', 16) - 12) / 48.0,
                (style.get('font_weight', 400) - 100) / 800.0,
                (style.get('line_height', 24) - 12) / 48.0
            ]
            styles.append(style_vec)
        
        return {
            'student': student_img,
            'raw': raw_img,
            'clip': clip_img,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'styles': torch.tensor(styles, dtype=torch.float32) if styles else torch.zeros(0, 8),
            'texts': texts,
            'image_id': idx,
            'source': sample.get('source', 'browser')
        }


def gate2_collate_fn(batch):
    """Collate function for Gate 2 browser dataset"""
    return {
        'student': torch.stack([b['student'] for b in batch]),
        'raw': torch.stack([b['raw'] for b in batch]),
        'clip': torch.stack([b['clip'] for b in batch]),
        'boxes': [b['boxes'] for b in batch],
        'labels': [b['labels'] for b in batch],
        'styles': [b['styles'] for b in batch],
        'texts': [b['texts'] for b in batch],
        'image_ids': [b['image_id'] for b in batch],
    }


def create_gate2_dataloader(config: dict, shuffle: bool = True, split: str = 'train') -> DataLoader:
    """Create Gate 2 dataloader with browser dataset"""
    dataset = Gate2BrowserDataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        collate_fn=gate2_collate_fn,
        pin_memory=False,
        drop_last=True
    )