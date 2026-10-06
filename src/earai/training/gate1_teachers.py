"""Gate 1 Teacher Ensemble - FasterRCNN + CLIP with correct preprocessing"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path


class Gate1TeacherEnsemble(nn.Module):
    """
    Teacher ensemble for Gate 1:
    - FasterRCNN: object detection (class + bbox)
    - CLIP: global semantic embedding
    
    Correct preprocessing for each teacher.
    """
    
    def __init__(self, device: str = "cuda", score_threshold: float = 0.70, max_objects: int = 16):
        super().__init__()
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.max_objects = max_objects
        
        # COCO category ID mapping (1-90 sparse -> 0-79 continuous)
        self.coco_mapping = {
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
        
        # Load teachers
        self.detector = self._load_detector()
        self.clip_model, self.clip_preprocess = self._load_clip()
        
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def _load_detector(self):
        """Load FasterRCNN detector"""
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load FasterRCNN detector: {e}")
    
    def _load_clip(self):
        """Load CLIP model"""
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            return model, preprocess
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP: {e}")
    
    @torch.no_grad()
    def forward(self, 
                student_images: torch.Tensor,      # [B, 3, H, W] ImageNet normalized
                raw_images: torch.Tensor,          # [B, 3, H, W] raw [0,1] for FasterRCNN
                clip_images: torch.Tensor) -> Dict: # [B, 3, 224, 224] CLIP preprocessed
        """
        Forward pass with separate inputs for each teacher.
        
        Args:
            student_images: ImageNet-normalized for EarAI student
            raw_images: Raw [0,1] for FasterRCNN
            clip_images: CLIP-preprocessed for CLIP
            
        Returns:
            Dict with detections, clip_embeddings
        """
        B = student_images.shape[0]
        H, W = student_images.shape[-2:]
        
        # FasterRCNN on raw images
        detections = self._detect(raw_images, H, W)
        
        # CLIP on CLIP-preprocessed images
        clip_emb = self._encode_clip(clip_images)
        
        return {
            'detections': detections,
            'clip_embeddings': clip_emb,
            'image_sizes': [(H, W)] * B
        }
    
    @torch.no_grad()
    def _detect(self, raw_images: torch.Tensor, H: int, W: int) -> List[Dict]:
        """Run FasterRCNN detection on raw [0,1] images"""
        # FasterRCNN expects raw [0,1] or [0,255] - we pass [0,1]
        results = self.detector(raw_images)
        
        detections = []
        for r in results:
            boxes = r['boxes']      # [N, 4] in pixel coordinates xyxy
            labels = r['labels']    # [N] COCO category IDs 1-90
            scores = r['scores']    # [N]
            
            # Filter by score
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            
            # Limit to max_objects
            if len(boxes) > self.max_objects:
                boxes = boxes[:self.max_objects]
                labels = labels[:self.max_objects]
                scores = scores[:self.max_objects]
            
            # Normalize boxes to [0,1] xyxy
            boxes_norm = boxes.clone()
            boxes_norm[:, [0, 2]] /= W
            boxes_norm[:, [1, 3]] /= H
            boxes_norm = boxes_norm.clamp(0, 1)
            
            # Map COCO labels to canonical 0-79
            labels_canonical = torch.tensor(
                [self.coco_mapping.get(int(l), 0) for l in labels],
                dtype=torch.long, device=self.device
            )
            
            detections.append({
                'boxes': boxes_norm.cpu().numpy(),      # [N, 4] normalized xyxy
                'labels': labels_canonical.cpu().numpy(), # [N] 0-79
                'scores': scores.cpu().numpy()          # [N]
            })
        
        return detections
    
    @torch.no_grad()
    def _encode_clip(self, clip_images: torch.Tensor) -> torch.Tensor:
        """Encode images with CLIP"""
        features = self.clip_model.encode_image(clip_images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features  # [B, 512]


class Gate1DataTransforms:
    """Creates the three image representations needed for Gate 1"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
        # Student: ImageNet normalization
        self.student_transform = T.Compose([
            T.ToTensor(),
            T.Resize(image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Raw for FasterRCNN: just resize, no normalization
        self.raw_transform = T.Compose([
            T.ToTensor(),
            T.Resize(image_size),
        ])
        
        # CLIP preprocessing - use the preprocess returned by clip.load()
        self.clip_preprocess = None  # Will be set after clip.load()
    
    def set_clip_preprocess(self, preprocess):
        self.clip_preprocess = preprocess
    
    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        image: HWC numpy array [0,255] uint8
        Returns dict with three tensor representations
        """
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        student_img = self.student_transform(pil_image)      # [3, H, W] ImageNet norm
        raw_img = self.raw_transform(pil_image)              # [3, H, W] raw [0,1]
        
        if self.clip_preprocess is not None:
            clip_img = self.clip_preprocess(pil_image)       # [3, 224, 224] CLIP norm
        else:
            # Fallback: resize + normalize
            clip_img = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
            ])(pil_image)
        
        return {
            'student': student_img,
            'raw': raw_img,
            'clip': clip_img
        }


def create_gate1_teachers(device: str = "cuda") -> Tuple[Gate1TeacherEnsemble, Gate1DataTransforms]:
    """Factory for Gate 1 teachers and transforms"""
    teachers = Gate1TeacherEnsemble(device)
    transforms = Gate1DataTransforms()
    # Set CLIP preprocess from loaded model
    transforms.set_clip_preprocess(teachers.clip_preprocess)
    return teachers, transforms