"""Gate 2 Teacher Ensemble - UI-specific teachers"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path


class Gate2TeacherEnsemble(nn.Module):
    """
    Teacher ensemble for Gate 2 (UI Understanding):
    - UI Object Detector (RT-DETR / FasterRCNN on UI data)
    - Layout Parser (LayoutXLM / DocLayout)
    - OCR (TrOCR / PaddleOCR)
    - Style Extractor (color, radius, font-size, weight)
    - CLIP (semantic embedding)
    """
    
    def __init__(self, device: str = "cpu", score_threshold: float = 0.5, max_objects: int = 50):
        super().__init__()
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.max_objects = max_objects
        
        # UI classes (15 classes)
        self.ui_classes = [
            'navbar', 'hero', 'section', 'card', 'button', 'input',
            'image', 'icon', 'heading', 'paragraph', 'badge',
            'modal', 'footer', 'container', 'link'
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.ui_classes)}
        
        # Load teachers
        self.ui_detector = self._load_ui_detector()
        self.layout_parser = self._load_layout_parser()
        self.ocr = self._load_ocr()
        self.style_extractor = self._load_style_extractor()
        self.clip_model, self.clip_preprocess = self._load_clip()
        
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def _load_ui_detector(self):
        """Load UI object detector - RT-DETR or FasterRCNN fine-tuned on UI data"""
        try:
            # Try to load a UI-specific detector, fallback to COCO FasterRCNN
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"UI detector load failed: {e}")
            return None
    
    def _load_layout_parser(self):
        """Load layout parser for hierarchy understanding"""
        try:
            # Use a simple segmentation model as proxy for layout
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Layout parser load failed: {e}")
            return None
    
    def _load_ocr(self):
        """Load OCR engine"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return pytesseract
        except Exception as e:
            print(f"OCR load failed: {e}")
            return None
    
    def _load_style_extractor(self):
        """Style extractor - extracts color, radius, font-size, weight from regions"""
        # Simple CNN for style prediction from image crops
        class StyleExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
                self.features = nn.Sequential(*list(backbone.children())[:-1])  # Remove fc
                self.regressor = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 8)  # bg_color(3), fg_color(3), radius(1), font_size(1)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.regressor(x)
        
        try:
            model = StyleExtractor()
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Style extractor load failed: {e}")
            return None
    
    def _load_clip(self):
        """Load CLIP for semantic embedding"""
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            return model, preprocess
        except Exception as e:
            print(f"CLIP load failed: {e}")
            return None, None
    
    @torch.no_grad()
    def detect_ui_objects(self, images: torch.Tensor) -> List[Dict]:
        """Detect UI objects using detector"""
        if self.ui_detector is None:
            return [{'boxes': [], 'labels': [], 'scores': []} for _ in range(len(images))]
        
        results = self.ui_detector(images)
        outputs = []
        for r in results:
            boxes = r['boxes'].cpu().numpy()
            labels = r['labels'].cpu().numpy()
            scores = r['scores'].cpu().numpy()
            
            # Filter by score
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            
            # Limit objects
            if len(boxes) > self.max_objects:
                boxes = boxes[:self.max_objects]
                labels = labels[:self.max_objects]
                scores = scores[:self.max_objects]
            
            # Normalize boxes to [0,1]
            H, W = images.shape[-2:]
            boxes_norm = boxes.copy()
            boxes_norm[:, [0, 2]] /= W
            boxes_norm[:, [1, 3]] /= H
            boxes_norm = np.clip(boxes_norm, 0, 1)
            
            outputs.append({
                'boxes': boxes_norm,      # [N, 4] normalized xyxy
                'labels': labels,         # [N] COCO class indices
                'scores': scores,         # [N]
            })
        return outputs
    
    @torch.no_grad()
    def parse_layout(self, images: torch.Tensor) -> List[Dict]:
        """Parse layout hierarchy using segmentation"""
        if self.layout_parser is None:
            return [{'hierarchy': [], 'containers': []} for _ in range(len(images))]
        
        # Use segmentation as proxy for layout containers
        with torch.no_grad():
            out = self.layout_parser(images)['out']
            seg_maps = out.argmax(dim=1).cpu().numpy()  # [B, H, W]
        
        outputs = []
        for b in range(len(images)):
            seg = seg_maps[b]
            # Find connected components as containers
            containers = self._seg_to_containers(seg)
            outputs.append({
                'containers': containers,
                'seg_map': seg
            })
        return outputs
    
    def _seg_to_containers(self, seg_map: np.ndarray) -> List[Dict]:
        """Convert segmentation map to container boxes"""
        containers = []
        unique_labels = np.unique(seg_map)
        for label in unique_labels:
            if label == 0:  # background
                continue
            mask = (seg_map == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                H, W = seg_map.shape
                containers.append({
                    'bbox': [x/W, y/H, (x+w)/W, (y+h)/H],
                    'class_id': int(label),
                    'area': float(area)
                })
        return containers
    
    @torch.no_grad()
    def extract_text(self, images: torch.Tensor) -> List[Dict]:
        """Extract text using OCR"""
        if self.ocr is None:
            return [{'text_regions': []} for _ in range(len(images))]
        
        outputs = []
        for i in range(len(images)):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            try:
                data = self.ocr.image_to_data(img_rgb, output_type='dict', lang='eng+rus')
                
                text_regions = []
                n = len(data['text'])
                h, w = img_np.shape[:2]
                
                for j in range(n):
                    text = data['text'][j].strip()
                    conf = int(data['conf'][j]) if data['conf'][j] != '-1' else 0
                    
                    if text and conf > 30:
                        x, y, bw, bh = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                        bbox = [x/w, y/h, (x+bw)/w, (y+bh)/h]
                        text_regions.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': conf / 100.0
                        })
                outputs.append({'text_regions': text_regions})
            except Exception as e:
                print(f"OCR error: {e}")
                outputs.append({'text_regions': []})
        
        return outputs
    
    @torch.no_grad()
    def extract_style(self, images: torch.Tensor, boxes: List[np.ndarray]) -> List[List[Dict]]:
        """Extract style properties (color, radius, font-size) for each box"""
        if self.style_extractor is None:
            return [[{'bg_color': [0.5, 0.5, 0.5], 'fg_color': [0, 0, 0], 'radius': 0, 'font_size': 16}] * len(b) for b in boxes]
        
        outputs = []
        for b, img_boxes in enumerate(boxes):
            img_styles = []
            for box in img_boxes:
                # Crop and resize box region
                x1, y1, x2, y2 = box
                H, W = images.shape[-2:]
                x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
                
                crop = images[b:b+1, :, y1:y2, x1:x2]
                if crop.shape[-1] < 8 or crop.shape[-2] < 8:
                    img_styles.append({'bg_color': [0.5, 0.5, 0.5], 'fg_color': [0, 0, 0], 'radius': 0, 'font_size': 16})
                    continue
                
                crop = torch.nn.functional.interpolate(crop, size=(64, 64), mode='bilinear', align_corners=False)
                
                with torch.no_grad():
                    style_vec = self.style_extractor(crop).squeeze().cpu().numpy()
                
                bg_color = style_vec[:3].tolist()
                fg_color = style_vec[3:6].tolist()
                radius = max(0, float(style_vec[6] * 50))
                font_size = max(8, float(style_vec[7] * 48 + 12))
                
                img_styles.append({
                    'bg_color': bg_color,
                    'fg_color': fg_color,
                    'radius': radius,
                    'font_size': font_size
                })
            outputs.append(img_styles)
        return outputs
    
    @torch.no_grad()
    def encode_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embeddings"""
        if self.clip_model is None:
            return torch.zeros(len(images), 512, device=self.device)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def forward(self, 
                student_images: torch.Tensor,      # [B, 3, H, W] ImageNet normalized
                raw_images: torch.Tensor,          # [B, 3, H, W] raw [0,1] for detector
                clip_images: torch.Tensor) -> Dict: # [B, 3, 224, 224] CLIP preprocessed
        """
        Full teacher forward pass for UI understanding.
        """
        B = student_images.shape[0]
        H, W = student_images.shape[-2:]
        
        # 1. UI Object Detection
        detections = self.detect_ui_objects(raw_images)
        
        # 2. Layout Parsing
        layout = self.parse_layout(raw_images)
        
        # 3. OCR
        text_outputs = self.extract_text(raw_images)
        
        # 4. Style Extraction (using detected boxes)
        all_boxes = [det['boxes'] for det in detections]
        styles = self.extract_style(raw_images, all_boxes)
        
        # 5. CLIP Embedding
        clip_emb = self.encode_clip(clip_images)
        
        # Combine all outputs
        targets = []
        for b in range(B):
            det = detections[b]
            lay = layout[b]
            txt = text_outputs[b]
            sty = styles[b]
            
            # Merge detections with style and text
            ui_elements = []
            for i in range(len(det['boxes'])):
                elem = {
                    'bbox': det['boxes'][i].tolist(),
                    'class_id': int(det['labels'][i]),
                    'class_name': 'unknown',  # Map COCO to UI classes later
                    'confidence': float(det['scores'][i]),
                    'style': sty[i] if i < len(sty) else {},
                    'text': ''  # Will match with OCR
                }
                ui_elements.append(elem)
            
            # Match OCR text to nearest element
            for text_region in text_outputs[b].get('text_regions', []):
                # Find closest element by bbox IoU
                best_iou = 0
                best_idx = -1
                text_box = text_region['bbox']
                for i, elem in enumerate(ui_elements):
                    iou = self._box_iou_single(text_box, elem['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx >= 0 and best_iou > 0.1:
                    ui_elements[best_idx]['text'] = text_region['text']
            
            targets.append({
                'image_id': b,
                'ui_elements': ui_elements,
                'containers': lay.get('containers', []),
                'clip_embedding': clip_emb[b].cpu().numpy()
            })
        
        return {
            'targets': targets,
            'clip_embeddings': clip_emb.cpu(),
            'image_sizes': [(H, W)] * B
        }
    
    def _box_iou_single(self, box1: List[float], box2: List[float]) -> float:
        """IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)


def create_gate2_teachers(device: str = "cpu") -> Tuple[Gate2TeacherEnsemble, object]:
    """Factory for Gate 2 teachers"""
    teachers = Gate2TeacherEnsemble(device)
    return teachers, None  # No separate OCR needed