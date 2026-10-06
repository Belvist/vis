"""Teacher models for EarAI distillation"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np


class TeacherEnsemble(nn.Module):
    """
    Ensemble of teacher models for EarAI distillation:
    - Object detection (YOLO/RT-DETR)
    - Semantic segmentation
    - OCR (Tesseract + transformer)
    - CLIP image encoder
    - Depth estimation (optional)
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        
        # Object detector - RT-DETR (fast, accurate)
        self.detector = self._load_detector()
        
        # CLIP for semantic understanding
        self.clip_model, self.clip_preprocess = self._load_clip()
        
        # Semantic segmentation
        self.segmenter = self._load_segmenter()
        
        # OCR - we'll use Tesseract externally + a transformer for reading order
        self.ocr_available = self._check_ocr()
        
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def _load_detector(self):
        """Load RT-DETR for object detection"""
        try:
            # Use torchvision's pretrained models or load from hub
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Detector load failed: {e}")
            return None
    
    def _load_clip(self):
        """Load CLIP for semantic understanding"""
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            return model, preprocess
        except Exception as e:
            print(f"CLIP load failed: {e}")
            return None, None
    
    def _load_segmenter(self):
        """Load semantic segmentation model"""
        try:
            # Use DeepLabV3 or similar
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Segmenter load failed: {e}")
            return None
    
    def _check_ocr(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
    
    @torch.no_grad()
    def detect_objects(self, images: torch.Tensor) -> List[Dict]:
        """
        Detect objects in batch of images.
        Returns list of dicts with boxes, labels, scores
        """
        if self.detector is None:
            return [{} for _ in range(len(images))]
        
        results = self.detector(images)
        outputs = []
        for r in results:
            outputs.append({
                'boxes': r['boxes'].cpu().numpy(),
                'labels': r['labels'].cpu().numpy(),
                'scores': r['scores'].cpu().numpy()
            })
        return outputs
    
    @torch.no_grad()
    def clip_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embeddings"""
        if self.clip_model is None:
            return torch.zeros(len(images), 512, device=self.device)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def clip_text_encode(self, texts: List[str]) -> torch.Tensor:
        """Get CLIP text embeddings"""
        if self.clip_model is None:
            return torch.zeros(len(texts), 512, device=self.device)
        
        import clip
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def segment(self, images: torch.Tensor) -> torch.Tensor:
        """Semantic segmentation"""
        if self.segmenter is None:
            return torch.zeros(len(images), 1, *images.shape[-2:], device=self.device)
        
        with torch.no_grad():
            out = self.segmenter(images)['out']
            return out.argmax(dim=1)
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict:
        """
        Full teacher forward pass.
        Returns all teacher predictions for distillation.
        """
        B = len(images)
        
        # Object detection
        detections = self.detect_objects(images)
        
        # CLIP embeddings
        clip_emb = self.clip_encode(images)
        
        # Segmentation
        seg_masks = self.segment(images)
        
        return {
            'detections': detections,
            'clip_embeddings': clip_emb,
            'segmentation': seg_masks,
            'batch_size': B
        }


class OCRTeacher:
    """OCR teacher using Tesseract + reading order transformer"""
    
    def __init__(self, languages: str = 'eng+rus'):
        self.languages = languages
        self.available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.available = True
            self.tesseract = pytesseract
        except:
            print("Tesseract not available")
    
    def process(self, image: np.ndarray) -> List[Dict]:
        """Run OCR on image, return text regions with bboxes"""
        if not self.available:
            return []
        
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data = self.tesseract.image_to_data(
                rgb, output_type='dict', lang='eng+rus'
            )
            
            regions = []
            n = len(data['text'])
            h, w = image.shape[:2]
            
            for i in range(n):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if text and conf > 30:
                    x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [x/w, y/h, (x+bw)/w, (y+bh)/h]
                    regions.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': conf / 100.0
                    })
            return regions
        except Exception as e:
            print(f"OCR error: {e}")
            return []


# Factory
def create_teachers(device: str = "cuda") -> Tuple[TeacherEnsemble, OCRTeacher]:
    """Create all teacher models"""
    teachers = TeacherEnsemble(device)
    ocr = OCRTeacher()
    return teachers, ocr