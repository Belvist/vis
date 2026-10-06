"""Foveated vision - peripheral + high-res crops"""
import cv2
import numpy as np
from typing import Optional, List
from ..core.config import EarAIConfig
from ..core.packets import VisionPacket, BBox


class FoveatedVision:
    """
    Implements foveated vision:
    - Peripheral: low-res full frame (256x256) every frame
    - Fovea: high-res crops (600x600) on demand
    """

    def __init__(self, config: EarAIConfig):
        self.config = config
        self.peripheral_size = config.peripheral_resolution
        self.fovea_size = config.fovea_resolution
        self.pending_fovea_requests: List[dict] = []

    def process_peripheral(self, frame: np.ndarray) -> np.ndarray:
        """Extract peripheral view - runs every frame"""
        # Resize to peripheral resolution
        peripheral = cv2.resize(frame, self.peripheral_size, interpolation=cv2.INTER_AREA)
        return peripheral

    def extract_fovea(self, frame: np.ndarray, request: dict) -> np.ndarray:
        """
        Extract high-res crop from original frame
        request: {"center": [cx, cy], "zoom": 3.0, "entity_id": 14}
        """
        h, w = frame.shape[:2]
        cx, cy = request["center"]
        zoom = request.get("zoom", 3.0)

        # Calculate crop size in original frame coordinates
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        # Center coordinates in pixels
        px = int(cx * w)
        py = int(cy * h)

        # Crop bounds
        x1 = max(0, px - crop_w // 2)
        y1 = max(0, py - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        # Extract and resize to fovea size
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((*self.fovea_size, 3), dtype=np.uint8)

        fovea = cv2.resize(crop, self.fovea_size, interpolation=cv2.INTER_CUBIC)
        return fovea

    def process_fovea_requests(self, frame: np.ndarray, packet: VisionPacket) -> VisionPacket:
        """Process pending fovea requests and add to packet"""
        fovea_crops = {}

        for req in self.pending_fovea_requests:
            fovea_img = self.extract_fovea(frame, req)
            entity_id = req.get("entity_id")
            fovea_crops[entity_id] = fovea_img

        # Add fovea crops to packet metadata (for native API)
        packet.fovea_crops = fovea_crops
        self.pending_fovea_requests.clear()
        return packet

    def add_fovea_request(self, request: dict):
        """Queue a fovea request for next frame"""
        self.pending_fovea_requests.append(request)

    def should_trigger_fovea(self, change_score: float, entity_confidence: float) -> bool:
        """Decide if fovea should be triggered"""
        return (change_score > self.config.fovea_trigger_threshold or
                entity_confidence > 0.85)

    def auto_fovea_requests(self, packet: VisionPacket) -> List[dict]:
        """Automatically generate fovea requests for interesting entities"""
        requests = []
        for entity in packet.entities:
            # High confidence objects
            if entity.confidence > 0.85:
                requests.append({
                    "type": "foveate",
                    "entity_id": entity.id,
                    "center": list(entity.bbox.center()),
                    "zoom": 3.0,
                    "priority": "high",
                    "reason": f"high_confidence_{entity.class_name}"
                })
            # Text regions
            if entity.class_name in ["text", "screen", "sign", "button"]:
                requests.append({
                    "type": "foveate",
                    "entity_id": entity.id,
                    "center": list(entity.bbox.center()),
                    "zoom": 4.0,
                    "priority": "high",
                    "reason": "text_region"
                })
            # Moving objects near center
            if (np.linalg.norm(entity.motion_vector) > 0.05 and
                0.3 < entity.bbox.center()[0] < 0.7 and
                0.3 < entity.bbox.center()[1] < 0.7):
                requests.append({
                    "type": "foveate",
                    "entity_id": entity.id,
                    "center": list(entity.bbox.center()),
                    "zoom": 2.0,
                    "priority": "normal",
                    "reason": "moving_center"
                })
        return requests