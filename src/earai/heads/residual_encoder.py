"""Residual encoder and ROI extraction for delta computation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ResidualROI:
    """Region of interest for residual computation"""
    bbox: np.ndarray          # [4] normalized x1,y1,x2,y2
    scale: int                # feature stride (4, 8, 16, 32)
    confidence: float
    source: str               # 'motion', 'change', 'uncertainty', 'fovea'


class ResidualEncoder(nn.Module):
    """
    Encodes residual regions (difference between warped prev frame and current).
    Lightweight encoder shared across scales.
    """
    
    def __init__(self, 
                 in_channels: int = 3,  # RGB residual
                 feature_dim: int = 256,
                 hidden_dim: int = 128):
        super().__init__()
        
        # Small encoder for residual patches
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(4, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
        
        # Also support feature-level residuals (difference in backbone features)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(4, feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, residual_patches: torch.Tensor) -> torch.Tensor:
        """
        residual_patches: [B, N, 3, H, W] - N residual crops
        Returns: [B, N, D] encoded features
        """
        B, N, C, H, W = residual_patches.shape
        patches = residual_patches.view(B * N, C, H, W)
        features = self.encoder(patches)
        return features.view(B, N, -1)
    
    def forward_features(self, feature_residuals: torch.Tensor) -> torch.Tensor:
        """
        feature_residuals: [B, N, D, h, w] - residual in feature space
        Returns: [B, N, D]
        """
        B, N, D, h, w = feature_residuals.shape
        feats = feature_residuals.view(B * N, D, h, w)
        encoded = self.feature_encoder(feats)
        return encoded.view(B, N, -1)


class ROIExtractor:
    """
    Extracts ROIs from frames based on:
    - Motion-compensated change detection
    - Uncertainty maps
    - Fovea requests
    - Text regions
    """
    
    def __init__(self, 
                 base_resolution: Tuple[int, int] = (256, 256),
                 roi_size: Tuple[int, int] = (64, 64),
                 context_margin: float = 0.2,
                 max_rois: int = 8):
        self.base_resolution = base_resolution
        self.roi_size = roi_size
        self.context_margin = context_margin
        self.max_rois = max_rois
    
    def extract_rois(self, 
                     frame: np.ndarray,
                     motion_transform,
                     prev_frame: Optional[np.ndarray],
                     change_map: Optional[np.ndarray],
                     uncertainty_map: Optional[np.ndarray],
                     fovea_requests: List[Dict],
                     text_regions: List[Dict]) -> List[ResidualROI]:
        """
        Extract prioritized ROIs for residual computation.
        Returns list of ROIs sorted by priority.
        """
        rois = []
        h, w = frame.shape[:2]
        
        # 1. Change-based ROIs (from change detector)
        if change_map is not None:
            change_rois = self._extract_change_rois(change_map, w, h)
            rois.extend(change_rois)
        
        # 2. Uncertainty-based ROIs
        if uncertainty_map is not None:
            unc_rois = self._extract_uncertainty_rois(uncertainty_map, w, h)
            rois.extend(unc_rois)
        
        # 3. Fovea requests (high priority)
        for req in fovea_requests:
            cx, cy = req['center']
            zoom = req.get('zoom', 3.0)
            roi = self._create_fovea_roi(cx, cy, zoom, w, h)
            rois.append(roi)
        
        # 4. Text regions
        for text in text_regions:
            bbox = text['bbox']
            roi = ResidualROI(
                bbox=np.array(bbox),
                scale=16,
                confidence=text.get('confidence', 0.5),
                source='text'
            )
            rois.append(roi)
        
        # Deduplicate and sort by priority
        rois = self._deduplicate_rois(rois)
        rois = sorted(rois, key=lambda r: -r.confidence)[:self.max_rois]
        
        return rois
    
    def _extract_change_rois(self, change_map: np.ndarray, w: int, h: int) -> List[ResidualROI]:
        """Extract ROIs from binary change map (already 0/255 uint8)"""
        rois = []
        # Find connected components
        # change_map is already 0/255 uint8 from compute_residual_frame
        change_uint8 = change_map.astype(np.uint8)
        contours, _ = cv2.findContours(change_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Increased min area to filter noise
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Smaller context margin
            margin_w = int(bw * 0.1)
            margin_h = int(bh * 0.1)
            x1 = max(0, x - margin_w) / w
            y1 = max(0, y - margin_h) / h
            x2 = min(w, x + bw + margin_w) / w
            y2 = min(h, y + bh + margin_h) / h
            
            roi = ResidualROI(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                scale=16,
                confidence=min(1.0, area / 1000.0),
                source='change'
            )
            rois.append(roi)
        
        return rois
    
    def _extract_uncertainty_rois(self, uncertainty_map: np.ndarray, w: int, h: int) -> List[ResidualROI]:
        """Extract ROIs from high uncertainty regions"""
        rois = []
        # Threshold uncertainty
        high_unc = (uncertainty_map > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(high_unc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            x1 = max(0, x) / w
            y1 = max(0, y) / h
            x2 = min(w, x + bw) / w
            y2 = min(h, y + bh) / h
            
            roi = ResidualROI(
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                scale=16,
                confidence=float(uncertainty_map[y:y+bh, x:x+bw].mean()),
                source='uncertainty'
            )
            rois.append(roi)
        
        return rois
    
    def _create_fovea_roi(self, cx: float, cy: float, zoom: float, w: int, h: int) -> ResidualROI:
        """Create ROI for foveated vision request"""
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        px, py = int(cx * w), int(cy * h)
        
        x1 = max(0, px - crop_w // 2) / w
        y1 = max(0, py - crop_h // 2) / h
        x2 = min(w, px + crop_w // 2) / w
        y2 = min(h, py + crop_h // 2) / h
        
        return ResidualROI(
            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
            scale=4,  # High-res
            confidence=1.0,
            source='fovea'
        )
    
    def _deduplicate_rois(self, rois: List[ResidualROI]) -> List[ResidualROI]:
        """Remove overlapping ROIs, keep highest confidence"""
        if not rois:
            return []
        
        # Sort by confidence
        rois = sorted(rois, key=lambda r: -r.confidence)
        
        keep = []
        for roi in rois:
            overlap = False
            for kept in keep:
                if self._iou(roi.bbox, kept.bbox) > 0.5:
                    overlap = True
                    break
            if not overlap:
                keep.append(roi)
        
        return keep
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def crop_rois(self, frame: np.ndarray, frame_warped: np.ndarray, rois: List[ResidualROI]) -> torch.Tensor:
        """
        Crop and prepare 9-channel patches from frame for residual encoder.
        Returns: [N, 9, H, W] tensor (current + warped_prev + diff)
        """
        patches = []
        h, w = frame.shape[:2]
        
        for roi in rois:
            x1, y1, x2, y2 = roi.bbox
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            
            # Crop from both frames
            crop_curr = frame[py1:py2, px1:px2]
            crop_warped = frame_warped[py1:py2, px1:px2]
            
            if crop_curr.size == 0 or crop_warped.size == 0:
                crop_curr = np.zeros((self.roi_size[1], self.roi_size[0], 3), dtype=np.uint8)
                crop_warped = np.zeros((self.roi_size[1], self.roi_size[0], 3), dtype=np.uint8)
            
            # Resize
            crop_curr = cv2.resize(crop_curr, self.roi_size, interpolation=cv2.INTER_AREA)
            crop_warped = cv2.resize(crop_warped, self.roi_size, interpolation=cv2.INTER_AREA)
            
            # Convert to RGB and normalize
            crop_curr = cv2.cvtColor(crop_curr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_warped = cv2.cvtColor(crop_warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Compute diff
            crop_diff = np.abs(crop_curr - crop_warped)
            
            # Stack: current (3) + warped_prev (3) + diff (3) = 9 channels
            patch_9ch = np.concatenate([crop_curr, crop_warped, crop_diff], axis=-1)
            patches.append(patch_9ch)
        
        if not patches:
            return torch.zeros((0, 9, *self.roi_size))
        
        batch = np.stack(patches)  # [N, H, W, 9]
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2)  # [N, 9, H, W]
        return batch


def compute_residual_frame(frame_curr: np.ndarray, 
                           frame_prev_warped: np.ndarray,
                           valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residual between current frame and motion-warped previous frame.
    Returns: (residual magnitude map [H, W], valid mask [H, W])
    """
    # Convert to grayscale first
    curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    prev_gray = cv2.cvtColor(frame_prev_warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # SIGNED difference (not absolute) - preserves direction of change
    signed_diff = curr_gray - prev_gray
    
    # Illumination compensation: local mean of signed diff
    kernel_size = 31
    local_mean = cv2.blur(signed_diff, (kernel_size, kernel_size))
    
    # Structural change = absolute of illumination-compensated difference
    structural = np.abs(signed_diff - local_mean)
    
    # Normalize
    magnitude = structural  # Already in [0, 1]
    
    # Threshold
    threshold = 0.05
    binary = (magnitude > threshold).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Apply valid mask if provided (to exclude warped borders)
    if valid_mask is not None:
        binary = cv2.bitwise_and(binary, (valid_mask > 0).astype(np.uint8) * 255)
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(binary)
    
    return magnitude, binary


def warp_frame(frame: np.ndarray, transform) -> Tuple[np.ndarray, np.ndarray]:
    """Warp frame using affine transform from previous to current frame.
    Returns: (warped frame, valid mask)
    """
    h, w = frame.shape[:2]
    # Convert normalized transform to pixel coordinates
    M = np.eye(3, dtype=np.float32)
    M[:2, :2] = transform.matrix
    M[:2, 2] = transform.translation * np.array([w, h])
    
    # Apply forward transform (prev -> current), NO WARP_INVERSE_MAP
    warped = cv2.warpAffine(frame, M[:2], (w, h), flags=cv2.INTER_LINEAR)
    
    # Create valid mask (areas that came from valid source pixels)
    valid_mask = np.ones((h, w), dtype=np.uint8)
    warped_mask = cv2.warpAffine(valid_mask, M[:2], (w, h), 
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return warped, warped_mask