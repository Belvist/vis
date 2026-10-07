"""Gate 2 Teacher - Uses precomputed browser/DOM ground truth (no COCO, no random style extractor)"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class Gate2Teacher(nn.Module):
    """
    Gate 2 Teacher - simply loads precomputed targets from browser/DOM cache.
    No COCO FasterRCNN, no random StyleExtractor.
    Browser provides: bbox, class, text, style, hierarchy from DOM.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        
        # UI classes (15 classes)
        self.ui_classes = [
            'navbar', 'hero', 'section', 'container', 'card', 'button', 'input',
            'image', 'icon', 'heading', 'paragraph', 'badge',
            'modal', 'footer', 'link'
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.ui_classes)}
        
        # CLIP for semantic embedding (only teacher we actually need at runtime)
        self.clip_model = None
        self.clip_preprocess = None
        self._load_clip()
        
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def _load_clip(self):
        """Load CLIP for semantic embedding"""
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            self.clip_model = model
            self.clip_preprocess = preprocess
        except Exception as e:
            raise RuntimeError(
                "Gate 2 requires the CLIP teacher. Install the gate2 extra "
                "(pip install -e '.[gate2]') before building the cache."
            ) from e
    
    @torch.no_grad()
    def encode_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embeddings"""
        if self.clip_model is None:
            raise RuntimeError("Gate 2 CLIP teacher is unavailable")
        
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def forward(self,
                student_images: torch.Tensor,      # [B, 3, H, W] ImageNet normalized
                raw_images: torch.Tensor,          # [B, 3, H, W] raw [0,1] - unused
                clip_images: torch.Tensor) -> Dict: # [B, 3, 224, 224] CLIP preprocessed
        """
        Forward pass - returns CLIP embeddings only.
        Targets (bbox, class, style, text, hierarchy) come from precomputed cache.
        """
        B = student_images.shape[0]
        
        # CLIP Embedding
        clip_emb = self.encode_clip(clip_images)
        
        # Return empty targets - actual targets come from cache
        targets = []
        for b in range(B):
            targets.append({
                'image_id': b,
                'ui_elements': [],  # Filled from cache
                'containers': [],
                'clip_embedding': clip_emb[b].cpu().numpy()
            })
        
        return {
            'targets': targets,
            'clip_embeddings': clip_emb.cpu(),
            'image_sizes': [(224, 224)] * B
        }


def create_gate2_teachers(device: str = "cpu") -> Tuple[Gate2Teacher, object]:
    """Factory for Gate 2 teachers - returns minimal teacher (CLIP only)"""
    teacher = Gate2Teacher(device)
    return teacher, None  # No separate OCR needed


# Backwards compatibility
Gate2TeacherEnsemble = Gate2Teacher