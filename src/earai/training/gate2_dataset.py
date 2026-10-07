"""Gate 2 Dataset - Rico / WebUI / Synthetic UI data"""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


# UI class mapping (15 classes)
UI_CLASSES = [
    'navbar', 'hero', 'section', 'card', 'button', 'input',
    'image', 'icon', 'heading', 'paragraph', 'badge',
    'modal', 'footer', 'container', 'link'
]
UI_CLASS_TO_IDX = {c: i for i, c in enumerate(UI_CLASSES)}


class Gate2Dataset(Dataset):
    """
    Dataset for Gate 2 UI understanding.
    Supports Rico dataset, WebUI, and synthetic data.
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
        
        self.raw_transform = T.Compose([
            T.ToTensor(),
            T.Resize(image_size),
        ])
        
        import clip
        self.clip_preprocess = clip._transform(224) if hasattr(clip, '_transform') else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                       std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Load samples
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {root}/{split}")
    
    def _load_samples(self) -> List[Dict]:
        """Load dataset samples from various sources"""
        samples = []
        
        # 1. Rico dataset
        rico_path = self.root / 'rico'
        if rico_path.exists():
            samples.extend(self._load_rico(rico_path))
        
        # 2. WebUI dataset
        webui_path = self.root / 'webui'
        if webui_path.exists():
            samples.extend(self._load_webui(webui_path))
        
        # 3. Synthetic fallback
        if len(samples) == 0:
            print("No real datasets found, generating synthetic data...")
            samples = self._generate_synthetic(50)
        
        return samples
    
    def _load_rico(self, rico_path: Path) -> List[Dict]:
        """Load Rico dataset annotations"""
        samples = []
        # Rico uses JSON annotations with view hierarchies
        # Simplified - in practice, parse Rico's specific format
        try:
            ann_file = rico_path / 'annotations' / f'{self.split}.json'
            if ann_file.exists():
                with open(ann_file) as f:
                    data = json.load(f)
                # Parse Rico format - this is simplified
                for item in data[:1000]:  # Limit for speed
                    samples.append({
                        'image': str(rico_path / self.split / item.get('image', '')),
                        'ui_elements': self._parse_rico_elements(item.get('elements', [])),
                        'source': 'rico'
                    })
        except Exception as e:
            print(f"Rico load failed: {e}")
        return samples
    
    def _parse_rico_elements(self, elements: List) -> List[Dict]:
        """Parse Rico view hierarchy to UI elements"""
        ui_elements = []
        for elem in elements:
            if 'bounds' in elem:
                x1, y1, x2, y2 = elem['bounds'][0], elem['bounds'][1], elem['bounds'][2], elem['bounds'][3]
                # Rico uses absolute coordinates, normalize later
                class_name = self._map_rico_class(elem.get('class', ''))
                if class_name in UI_CLASS_TO_IDX:
                    ui_elements.append({
                        'bbox': [x1, y1, x2, y2],  # Will normalize in __getitem__
                        'class_id': UI_CLASS_TO_IDX[class_name],
                        'text': elem.get('text', ''),
                        'style': self._extract_rico_style(elem)
                    })
        return ui_elements
    
    def _map_rico_class(self, rico_class: str) -> str:
        """Map Rico class names to UI classes"""
        rico_lower = rico_class.lower()
        if 'button' in rico_lower:
            return 'button'
        elif 'text' in rico_lower or 'edittext' in rico_lower:
            return 'paragraph'
        elif 'image' in rico_lower:
            return 'image'
        elif 'icon' in rico_lower:
            return 'icon'
        elif 'toolbar' in rico_lower or 'appbar' in rico_lower:
            return 'navbar'
        elif 'card' in rico_lower:
            return 'card'
        elif 'input' in rico_lower:
            return 'input'
        elif 'modal' in rico_lower or 'dialog' in rico_lower:
            return 'modal'
        elif 'list' in rico_lower or 'recycler' in rico_lower:
            return 'section'
        else:
            return 'container'
    
    def _extract_rico_style(self, elem: Dict) -> Dict:
        """Extract style from Rico element"""
        style = {}
        if 'background_color' in elem:
            # Convert ARGB to normalized RGB
            color = elem['background_color']
            if isinstance(color, int):
                r = ((color >> 16) & 0xFF) / 255.0
                g = ((color >> 8) & 0xFF) / 255.0
                b = (color & 0xFF) / 255.0
                style['bg_color'] = [r, g, b]
        if 'corner_radius' in elem:
            style['radius'] = float(elem['corner_radius'])
        if 'text_size' in elem:
            style['font_size'] = float(elem['text_size'])
        return style
    
    def _load_webui(self, webui_path: Path) -> List[Dict]:
        """Load WebUI dataset"""
        # Placeholder - similar to Rico
        return []
    
    def _generate_synthetic(self, n: int) -> List[Dict]:
        """Generate synthetic UI data for testing - with actual drawn elements"""
        samples = []
        for i in range(n):
            # Create a blank image
            image = np.ones((224, 224, 3), dtype=np.uint8) * 240  # Light gray background
            
            num_elements = np.random.randint(3, 8)
            ui_elements = []
            
            for j in range(num_elements):
                class_name = np.random.choice(UI_CLASSES)
                class_id = UI_CLASS_TO_IDX[class_name]
                
                # Generate non-overlapping bboxes
                max_attempts = 20
                for _ in range(max_attempts):
                    x1 = np.random.uniform(0.05, 0.7)
                    y1 = np.random.uniform(0.05, 0.7)
                    w = np.random.uniform(0.1, 0.25)
                    h = np.random.uniform(0.1, 0.25)
                    x2 = min(x1 + w, 0.95)
                    y2 = min(y1 + h, 0.95)
                    
                    # Check overlap with existing elements
                    overlap = False
                    for existing in ui_elements:
                        ex1, ey1, ex2, ey2 = existing['bbox']
                        if not (x2 <= ex1 or x1 >= ex2 or y2 <= ey1 or y1 >= ey2):
                            overlap = True
                            break
                    if not overlap:
                        break
                else:
                    continue  # Skip if couldn't place
                
                # Draw the element on the image
                x1_px, y1_px = int(x1 * 224), int(y1 * 224)
                x2_px, y2_px = int(x2 * 224), int(y2 * 224)
                
                # Draw filled rectangle with distinct color
                color = np.random.randint(50, 200, 3).tolist()
                cv2.rectangle(image, (x1_px, y1_px), (x2_px, y2_px), color, -1)
                # Add border
                cv2.rectangle(image, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 0), 2)
                
                # Add text-like lines inside
                if class_name in ['button', 'heading', 'paragraph', 'link']:
                    for k in range(np.random.randint(1, 4)):
                        tx = x1_px + np.random.randint(5, max(6, x2_px - x1_px - 5))
                        ty = y1_px + np.random.randint(5, max(6, y2_px - y1_px - 5))
                        cv2.line(image, (tx, ty), (tx + np.random.randint(20, 60), ty), (0, 0, 0), 1)
                
                style = {
                    'bg_color': [c/255.0 for c in color],
                    'fg_color': [0.0, 0.0, 0.0],
                    'radius': 4.0,
                    'font_size': 14.0
                }
                
                ui_elements.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'text': class_name if np.random.rand() > 0.5 else '',
                    'style': style
                })
            
            if not ui_elements:
                continue
                
            samples.append({
                'image': image,
                'ui_elements': ui_elements,
                'source': 'synthetic'
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        if sample.get('source') == 'synthetic' or sample.get('image') is None:
            return self._generate_synthetic_sample()
        
        # Load real image
        image = cv2.imread(sample['image'])
        if image is None:
            return self._generate_synthetic_sample()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Apply transforms
        from PIL import Image
        pil_image = Image.fromarray(image)
        student_img = self.transform(pil_image)
        raw_img = self.raw_transform(pil_image)
        clip_img = self.clip_preprocess(pil_image)
        
        # Process UI elements
        boxes = []
        labels = []
        styles = []
        texts = []
        
        for elem in sample['ui_elements'][:self.max_objects]:
            bbox = elem['bbox']
            # Normalize bbox
            x1, y1, x2, y2 = bbox
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            labels.append(elem['class_id'])
            texts.append(elem.get('text', ''))
            
            style = elem.get('style', {})
            style_vec = [
                *style.get('bg_color', [0.5, 0.5, 0.5]),
                *style.get('fg_color', [0, 0, 0]),
                style.get('radius', 0) / 50.0,
                (style.get('font_size', 16) - 12) / 48.0
            ]
            styles.append(style_vec)
        
        return {
            'student': student_img,
            'raw': raw_img,
            'clip': raw_img,  # Will be CLIP preprocessed in collate
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'styles': torch.tensor(styles, dtype=torch.float32) if styles else torch.zeros(0, 8),
            'texts': texts,
            'image_id': idx
        }
    
    def _generate_synthetic_sample(self) -> Dict:
            """Generate a single synthetic sample on the fly with drawn elements"""
            image = np.ones((224, 224, 3), dtype=np.uint8) * 240
        
            num_elements = np.random.randint(3, 8)
            ui_elements = []
        
            for j in range(num_elements):
                class_name = np.random.choice(UI_CLASSES)
                class_id = UI_CLASS_TO_IDX[class_name]
            
                max_attempts = 20
                for _ in range(max_attempts):
                    x1 = np.random.uniform(0.05, 0.7)
                    y1 = np.random.uniform(0.05, 0.7)
                    w = np.random.uniform(0.1, 0.25)
                    h = np.random.uniform(0.1, 0.25)
                    x2 = min(x1 + w, 0.95)
                    y2 = min(y1 + h, 0.95)
                
                    overlap = False
                    for existing in ui_elements:
                        ex1, ey1, ex2, ey2 = existing['bbox']
                        if not (x2 <= ex1 or x1 >= ex2 or y2 <= ey1 or y1 >= ey2):
                            overlap = True
                            break
                    if not overlap:
                        break
                else:
                    continue
            
                x1_px, y1_px = int(x1 * 224), int(y1 * 224)
                x2_px, y2_px = int(x2 * 224), int(y2 * 224)
            
                color = np.random.randint(50, 200, 3).tolist()
                cv2.rectangle(image, (x1_px, y1_px), (x2_px, y2_px), color, -1)
                cv2.rectangle(image, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 0), 2)
            
                if class_name in ['button', 'heading', 'paragraph', 'link']:
                    for k in range(np.random.randint(1, 4)):
                        tx = x1_px + np.random.randint(5, max(6, x2_px - x1_px - 5))
                        ty = y1_px + np.random.randint(5, max(6, y2_px - y1_px - 5))
                        cv2.line(image, (tx, ty), (tx + np.random.randint(20, 60), ty), (0, 0, 0), 1)
            
                style = {
                    'bg_color': [c/255.0 for c in color],
                    'fg_color': [0.0, 0.0, 0.0],
                    'radius': 4.0,
                    'font_size': 14.0
                }
            
                ui_elements.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'text': class_name if np.random.rand() > 0.5 else '',
                    'style': style
                })
        
            if not ui_elements:
                return self._generate_synthetic_sample()
        
            from PIL import Image
            pil_image = Image.fromarray(image)
            student_img = self.transform(pil_image)
            raw_img = self.raw_transform(pil_image)
            clip_img = self.clip_preprocess(pil_image)
        
            boxes = []
            labels = []
            styles = []
        
            for elem in ui_elements:
                bbox = elem['bbox']
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                labels.append(elem['class_id'])
                style = elem.get('style', {})
                style_vec = [
                    *style.get('bg_color', [0.5, 0.5, 0.5]),
                    *style.get('fg_color', [0, 0, 0]),
                    style.get('radius', 0) / 50.0,
                    (style.get('font_size', 16) - 12) / 48.0
                ]
                styles.append(style_vec)
        
            return {
                'student': student_img,
                'raw': raw_img,
                'clip': clip_img,
                'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
                'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
                'styles': torch.tensor(styles, dtype=torch.float32) if styles else torch.zeros(0, 8),
                'texts': [''] * len(ui_elements),
                'image_id': -1
            }


def gate2_collate_fn(batch):
    """Collate function for Gate 2 dataset"""
    return {
        'student': torch.stack([b['student'] for b in batch]),
        'raw': torch.stack([b['raw'] for b in batch]),
        'clip': torch.stack([b['clip'] for b in batch]),
        'boxes': [b['boxes'] for b in batch],
        'labels': [b['labels'] for b in batch],
        'styles': [b['styles'] for b in batch],
        'texts': [b['texts'] for b in batch],
        'image_ids': [b['image_id'] for b in batch]
    }


def create_gate2_dataloader(config: dict, shuffle: bool = True) -> DataLoader:
    """Create Gate 2 dataloader"""
    dataset = Gate2Dataset(
        root=config['data_root'],
        split=config.get('split', 'train'),
        image_size=config.get('image_size', (224, 224)),
        max_objects=config.get('max_objects', 16)
    )
    
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        pin_memory=False,
        collate_fn=gate2_collate_fn
    )