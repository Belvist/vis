"""Lightweight change detector - runs every frame at 30fps"""
import numpy as np
import cv2
from typing import Optional
from ..core.config import VeraEyeConfig


class ChangeDetector:
    """
    Ultra-fast change detection using:
    1. Pixel difference (grayscale)
    2. Optical flow magnitude (sparse)
    3. Histogram difference
    All designed to run in <1ms per frame
    """

    def __init__(self, config: VeraEyeConfig):
        self.config = config
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.frame_count = 0

        # Optical flow params (sparse, fast)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

    def detect(self, frame: np.ndarray) -> dict:
        """
        Returns change metrics for current frame
        frame: BGR uint8 [H, W, 3]
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Resize to peripheral resolution for speed
        small = cv2.resize(frame, self.config.peripheral_resolution)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        result = {
            "pixel_change": 0.0,
            "flow_magnitude": 0.0,
            "hist_change": 0.0,
            "changed_regions": [],
            "significant_change": False
        }

        if self.prev_gray is not None:
            # 1. Pixel difference (normalized)
            diff = cv2.absdiff(gray, self.prev_gray)
            pixel_change = np.mean(diff) / 255.0
            result["pixel_change"] = float(pixel_change)

            # 2. Sparse optical flow magnitude
            if self.prev_keypoints is not None and len(self.prev_keypoints) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_keypoints, None, **self.lk_params
                )
                if next_pts is not None:
                    valid = status.flatten() == 1
                    if np.any(valid):
                        displacement = next_pts[valid] - self.prev_keypoints[valid]
                        flow_mag = np.mean(np.linalg.norm(displacement, axis=1))
                        result["flow_magnitude"] = float(flow_mag / max(h, w))  # normalized

            # 3. Histogram difference
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if self.prev_hist is not None:
                hist_change = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                result["hist_change"] = float(hist_change)

            # 4. Detect changed regions (thresholded diff)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 50:  # Minimum region size
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Normalize to 0-1
                    result["changed_regions"].append({
                        "bbox": [x / self.config.peripheral_resolution[0],
                                 y / self.config.peripheral_resolution[1],
                                 (x + w) / self.config.peripheral_resolution[0],
                                 (y + h) / self.config.peripheral_resolution[1]],
                        "area": float(area)
                    })

            # Overall significance
            result["significant_change"] = (
                pixel_change > self.config.change_detection_threshold or
                result["flow_magnitude"] > 0.01 or
                result["hist_change"] > 0.1
            )

        # Update previous frame
        self.prev_gray = gray.copy()
        self.prev_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        self.prev_hist = cv2.normalize(self.prev_hist, self.prev_hist).flatten()

        # Detect new keypoints for next frame
        self.prev_keypoints = cv2.goodFeaturesToTrack(gray, **self.feature_params)

        return result

    def reset(self):
        self.prev_gray = None
        self.prev_hist = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.frame_count = 0