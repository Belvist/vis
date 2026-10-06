"""Motion compensation - global affine estimation for camera motion"""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MotionTransform:
    """Affine transform: x' = M @ x + t"""
    matrix: np.ndarray  # 2x2
    translation: np.ndarray  # 2
    confidence: float
    
    def warp_points(self, points: np.ndarray) -> np.ndarray:
        """Warp points: (N, 2) -> (N, 2)"""
        return (points @ self.matrix.T) + self.translation
    
    def warp_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """Warp bbox [x1, y1, x2, y2] normalized"""
        corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]
        ])
        warped = self.warp_points(corners)
        x1, y1 = warped.min(axis=0)
        x2, y2 = warped.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def inverse(self) -> "MotionTransform":
        """Inverse transform"""
        inv_M = np.linalg.inv(self.matrix)
        inv_t = -inv_M @ self.translation
        return MotionTransform(inv_M, inv_t, self.confidence)


class MotionEstimator:
    """
    Estimate global camera motion between frames using sparse optical flow + RANSAC.
    Runs on downsampled frames for speed.
    """
    
    def __init__(self, 
                 downsample: int = 4,
                 max_features: int = 200,
                 min_features: int = 20,
                 ransac_threshold: float = 3.0):
        self.downsample = downsample
        self.max_features = max_features
        self.min_features = min_features
        self.ransac_threshold = ransac_threshold
        
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        
        # LK optical flow params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
    
    def estimate(self, frame: np.ndarray) -> Optional[MotionTransform]:
        """
        Estimate global motion from previous frame to current.
        Returns None if not enough features or first frame.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // self.downsample, h // self.downsample))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.prev_pts is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return None
        
        # Optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        if next_pts is None:
            self._reset(gray)
            return None
        
        # Filter valid
        valid = status.flatten() == 1
        if np.sum(valid) < self.min_features:
            self._reset(gray)
            return None
        
        prev_valid = self.prev_pts[valid].reshape(-1, 2)
        next_valid = next_pts[valid].reshape(-1, 2)
        
        # Estimate affine with RANSAC
        try:
            M, inliers = cv2.estimateAffinePartial2D(
                prev_valid, next_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=2000,
                confidence=0.99
            )
        except cv2.error:
            self._reset(gray)
            return None
        
        if M is None or inliers is None or np.sum(inliers) < self.min_features:
            self._reset(gray)
            return None
        
        # M is 2x3: [A|t] where A is 2x2, t is 2x1
        matrix = M[:, :2].astype(np.float32)
        translation = M[:, 2].astype(np.float32)
        
        # Normalize translation to 0-1 coords
        translation = translation / np.array([w // self.downsample, h // self.downsample], dtype=np.float32)
        
        confidence = float(np.sum(inliers)) / len(valid)
        
        # Update for next frame
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        
        return MotionTransform(matrix, translation, confidence)
    
    def _reset(self, gray: np.ndarray):
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
    
    def reset(self):
        self.prev_gray = None
        self.prev_pts = None


class FlowMotionEstimator:
    """
    Alternative: dense optical flow for more accurate motion (slower).
    Uses Farneback on small resolution.
    """
    
    def __init__(self, downsample: int = 8):
        self.downsample = downsample
        self.prev_gray: Optional[np.ndarray] = None
    
    def estimate(self, frame: np.ndarray) -> Optional[MotionTransform]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // self.downsample, h // self.downsample))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Dense flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Median flow = global motion
        med_y, med_x = np.median(flow, axis=(0, 1))
        
        # Normalize
        translation = np.array([med_x, med_y], dtype=np.float32) / np.array(
            [w // self.downsample, h // self.downsample], dtype=np.float32
        )
        
        # Simple translation matrix
        matrix = np.eye(2, dtype=np.float32)
        
        self.prev_gray = gray
        
        return MotionTransform(matrix, translation, 0.8)
    
    def reset(self):
        self.prev_gray = None