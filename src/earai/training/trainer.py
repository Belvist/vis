"""Training loop and dataset for EarAI"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import cv2
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm


class EarAIDataset(Dataset):
    """
    Dataset for EarAI training.
    Supports COCO, custom datasets, synthetic data.
    """
    
    def __init__(self, 
                 root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (224, 224),
                 max_objects: int = 16):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        
        # Transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load dataset samples"""
        samples = []
        
        # Look for COCO format
        ann_file = self.root / f'annotations/instances_{self.split}.json'
        if ann_file.exists():
            return self._load_coco(ann_file)
        
        # Look for flat image directory
        img_dir = self.root / self.split / 'images'
        if img_dir.exists():
            for img_path in img_dir.glob('*.jpg'):
                samples.append({'image': str(img_path), 'annotations': []})
            return samples
        
        # Synthetic fallback
        return self._generate_synthetic(1000)
    
    def _load_coco(self, ann_file: Path) -> List[Dict]:
        """Load COCO annotations"""
        with open(ann_file) as f:
            coco = json.load(f)
        
        images = {img['id']: img for img in coco['images']}
        anns_by_img = {}
        for ann in coco['annotations']:
            anns_by_img.setdefault(ann['image_id'], []).append(ann)
        
        samples = []
        for img_id, img_info in images.items():
            img_path = self.root / self.split / 'images' / img_info['file_name']
            if img_path.exists():
                samples.append({
                    'image': str(img_path),
                    'annotations': anns_by_img.get(img_id, []),
                    'image_id': img_id
                })
        return samples
    
    def _generate_synthetic(self, n: int) -> List[Dict]:
        """Generate synthetic samples for testing"""
        samples = []
        for i in range(n):
            samples.append({
                'image': None,  # Will generate on the fly
                'annotations': [],
                'synthetic': True
            })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        if sample.get('synthetic'):
            return self._generate_synthetic_sample()
        
        # Load image
        image = cv2.imread(sample['image'])
        if image is None:
            return self._generate_synthetic_sample()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Apply transform
        image_tensor = self.transform(image)
        
        # Process annotations
        boxes = []
        labels = []
        for ann in sample.get('annotations', [])[:self.max_objects]:
            if 'bbox' in ann:
                x, y, bw, bh = ann['bbox']
                boxes.append([x/w, y/h, (x+bw)/w, (y+bh)/h])  # normalized
                labels.append(ann.get('category_id', 1))
        
        return {
            'image': image_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': sample.get('image_id', idx)
        }
    
    def _generate_synthetic_sample(self) -> Dict:
        """Generate a synthetic training sample"""
        # Random background
        image = np.random.randint(0, 255, (*self.image_size[::-1], 3), dtype=np.uint8)
        
        # Add random shapes
        num_shapes = np.random.randint(1, 5)
        boxes = []
        labels = []
        
        for i in range(num_shapes):
            x1 = np.random.randint(0, self.image_size[0] - 20)
            y1 = np.random.randint(0, self.image_size[1] - 20)
            x2 = np.random.randint(x1 + 10, min(x1 + 60, self.image_size[0]))
            y2 = np.random.randint(y1 + 10, min(y1 + 60, self.image_size[1]))
            
            color = np.random.randint(0, 255, 3).tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            
            boxes.append([x1/self.image_size[0], y1/self.image_size[1], 
                         x2/self.image_size[0], y2/self.image_size[1]])
            labels.append(np.random.randint(1, 80))
        
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': -1
        }


def earai_collate_fn(batch):
    """Custom collate function for variable number of boxes"""
    images = torch.stack([b['image'] for b in batch])
    boxes = [b['boxes'] for b in batch]
    labels = [b['labels'] for b in batch]
    image_ids = [b['image_id'] for b in batch]
    
    return {
        'image': images,
        'boxes': boxes,  # List of tensors [N_i, 4]
        'labels': labels,  # List of tensors [N_i]
        'image_id': image_ids
    }


class EarAITrainer:
    """
    Main training loop for EarAI.
    Handles teacher-student distillation, logging, checkpointing.
    """
    
    def __init__(self,
                 student_model: nn.Module,
                 decoder: nn.Module,
                 teachers,
                 losses: Dict[str, nn.Module],
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 log_dir: str = './logs',
                 checkpoint_dir: str = './checkpoints',
                 ocr_teacher=None):
        
        self.student = student_model.to(device)
        self.decoder = decoder.to(device)
        self.teachers = teachers.to(device) if hasattr(teachers, 'to') else teachers
        self.ocr_teacher = ocr_teacher
        self.losses = {k: v.to(device) for k, v in losses.items()}
        self.optimizer = optimizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
        self.epoch = 0
        
        # Logging
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except:
            pass
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.student.train()
        self.decoder.train()
        
        images = batch['images'].to(self.device)
        B = images.shape[0]
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teachers(images)
            
            # Also get OCR for text regions
            if self.ocr_teacher and self.ocr_teacher.available:
                ocr_results = []
                for b in range(B):
                    img_np = images[b].permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    ocr_results.append(self.ocr_teacher.process(img_np))
                teacher_out['ocr'] = ocr_results
        
        # Student forward
        student_tokens = self.student(images)  # [B, N, D]
        student_out = self.decoder(student_tokens)
        
        # Compute losses
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            if name == 'distillation':
                l = loss_fn(student_tokens, teacher_out, student_out)
            elif name == 'contrastive':
                # Need text embeddings from teacher
                text_emb = teacher_out.get('clip_text_embeddings')
                if text_emb is not None:
                    l = loss_fn(student_tokens.mean(1), text_emb)
                else:
                    continue
            elif name == 'state_transition':
                # State transition loss - needs previous state and residual
                # For now skip - requires streaming context
                continue
            else:
                continue
            
            for k, v in l.items():
                if k == 'total':
                    if isinstance(v, torch.Tensor):
                        total_loss = total_loss + v
                    else:
                        total_loss = total_loss + torch.tensor(v, device=self.device, dtype=torch.float32)
                loss_dict[f'{name}/{k}'] = v.item() if isinstance(v, torch.Tensor) else v
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.student.parameters()) + list(self.decoder.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        loss_dict['total'] = total_loss.item()
        self.step += 1
        
        # Logging
        if self.step % 100 == 0 and self.writer:
            for k, v in loss_dict.items():
                self.writer.add_scalar(f'train/{k}', v, self.step)
        
        return loss_dict
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = {}
        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch}')
        
        for batch in pbar:
            # Batch is already collated by DataLoader
            batch_dict = {
                'images': batch['image'].to(self.device) if isinstance(batch, dict) else torch.stack([b['image'] for b in batch]).to(self.device)
            }
            
            losses = self.train_step(batch_dict)
            
            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v)
            
            pbar.set_postfix({k: v for k, v in losses.items() if 'total' in k or 'bbox' in k})
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        if self.writer:
            for k, v in avg_losses.items():
                self.writer.add_scalar(f'epoch/{k}', v, self.epoch)
        
        self.epoch += 1
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.student.eval()
        self.decoder.eval()
        
        val_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device) if isinstance(batch, dict) else torch.stack([b['image'] for b in batch]).to(self.device)
                batch_dict = {'images': images}
                
                # Teacher forward
                teacher_out = self.teachers(images)
                
                # Student forward
                student_tokens = self.student(images)
                student_out = self.decoder(student_tokens)
                
                # Compute losses
                for name, loss_fn in self.losses.items():
                    if name == 'distillation':
                        l = loss_fn(student_tokens, teacher_out, student_out)
                        for k, v in l.items():
                            val_losses.setdefault(f'{name}/{k}', []).append(
                                v.item() if isinstance(v, torch.Tensor) else v
                            )
        
        return {k: np.mean(v) for k, v in val_losses.items()}
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save model checkpoint"""
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save({
            'step': self.step,
            'epoch': self.epoch,
            'student_state': self.student.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        print(f'Checkpoint saved: {path}')
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt['student_state'])
        self.decoder.load_state_dict(ckpt['decoder_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.step = ckpt['step']
        self.epoch = ckpt['epoch']
        print(f'Checkpoint loaded: {path}')


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders"""
    train_dataset = EarAIDataset(
        config['data_root'],
        split='train',
        image_size=config.get('image_size', (224, 224))
    )
    val_dataset = EarAIDataset(
        config['data_root'],
        split='val',
        image_size=config.get('image_size', (224, 224))
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=earai_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=earai_collate_fn
    )
    
    return train_loader, val_loader


def create_trainer(config: dict, student, decoder, teachers, ocr_teacher=None):
    """Factory for trainer"""
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(decoder.parameters()),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Losses
    from earai.training.losses import create_losses
    losses = create_losses(config.get('losses', {}))
    
    trainer = EarAITrainer(
        student_model=student,
        decoder=decoder,
        teachers=teachers,
        losses=losses,
        optimizer=optimizer,
        device=config.get('device', 'cuda'),
        log_dir=config.get('log_dir', './logs'),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        ocr_teacher=ocr_teacher
    )
    
    return trainer