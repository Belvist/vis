"""Core data structures for EarAI vision packets"""
from dataclasses import dataclass, field
from typing import Literal, Optional
import time
import uuid
import numpy as np


@dataclass
class BBox:
    """Normalized bounding box [x1, y1, x2, y2] in 0-1 range"""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_xywh(self) -> tuple:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (self.x2 - self.x1) * (self.y2 - self.y1)
        area2 = (other.x2 - other.x1) * (other.y2 - other.y1)
        return inter / (area1 + area2 - inter)

    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class Entity:
    """Tracked entity in scene memory"""
    id: int
    bbox: BBox
    visual_embedding: np.ndarray  # 256-d
    semantic_embedding: np.ndarray  # 256-d
    motion_vector: np.ndarray  # [dx, dy] normalized
    class_name: str
    confidence: float
    first_seen: float
    last_seen: float
    frames_tracked: int = 0
    is_occluded: bool = False
    occlusion_frames: int = 0
    metadata: dict = field(default_factory=dict)

    def age(self, current_time: float) -> float:
        return current_time - self.last_seen

    def is_stale(self, current_time: float, max_age: float = 5.0) -> bool:
        return self.age(current_time) > max_age


@dataclass
class TextRegion:
    """Detected text region"""
    value: str
    bbox: BBox
    confidence: float
    language: Optional[str] = None


@dataclass
class SceneChange:
    """Detected change in scene"""
    type: Literal["appeared", "disappeared", "moved", "changed", "occluded"]
    entity_id: Optional[int] = None
    bbox: Optional[BBox] = None
    details: str = ""


@dataclass
class VisionPacket:
    """Main output packet - sent to LLM/agent"""
    t: float  # timestamp
    frame_id: int

    # Global scene embedding (256-d)
    scene_embedding: np.ndarray

    # Tracked entities
    entities: list[Entity]

    # Detected text
    text_regions: list[TextRegion]

    # Changes since last packet
    changes: list[SceneChange]

    # Foveated crops requested
    fovea_requests: list[dict] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    inference_mode: Literal["full", "tracking_only", "change_triggered"] = "full"
    peripheral_resolution: tuple = (256, 256)

    def to_universal_api(self) -> dict:
        """JSON-serializable format for any LLM"""
        return {
            "t": self.t,
            "frame_id": self.frame_id,
            "scene_summary": self._scene_summary(),
            "entities": [
                {
                    "id": e.id,
                    "class": e.class_name,
                    "bbox": [e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2],
                    "confidence": round(e.confidence, 3),
                    "position": "left" if e.bbox.center()[0] < 0.33 else "center" if e.bbox.center()[0] < 0.66 else "right",
                    "distance": "near" if e.bbox.area() > 0.1 else "far",
                    "motion": "stationary" if np.linalg.norm(e.motion_vector) < 0.01 else "moving",
                    "holding": e.metadata.get("holding", None)
                }
                for e in self.entities
            ],
            "text": [
                {"value": t.value, "bbox": [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2]}
                for t in self.text_regions
            ],
            "changes": [
                {"type": c.type, "entity_id": c.entity_id, "details": c.details}
                for c in self.changes
            ],
            "fovea_requests": self.fovea_requests
        }

    def _scene_summary(self) -> str:
        parts = []
        if self.entities:
            parts.append(f"{len(self.entities)} objects: " + ", ".join(f"{e.class_name}(#{e.id})" for e in self.entities[:5]))
        if self.text_regions:
            parts.append(f"Text: {', '.join(t.value[:30] for t in self.text_regions[:3])}")
        if self.changes:
            parts.append(f"Changes: {', '.join(c.type + (f' #{c.entity_id}' if c.entity_id else '') for c in self.changes)}")
        return "; ".join(parts) if parts else "empty scene"

    def to_native_api(self) -> dict:
        """Rich format for native model with embeddings"""
        return {
            "t": self.t,
            "frame_id": self.frame_id,
            "scene_embedding": self.scene_embedding.tolist(),
            "entities": [
                {
                    "id": e.id,
                    "bbox": [e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2],
                    "visual_embedding": e.visual_embedding.tolist(),
                    "semantic_embedding": e.semantic_embedding.tolist(),
                    "motion_vector": e.motion_vector.tolist(),
                    "class_name": e.class_name,
                    "confidence": e.confidence,
                    "frames_tracked": e.frames_tracked,
                    "metadata": e.metadata
                }
                for e in self.entities
            ],
            "text_regions": [
                {
                    "value": t.value,
                    "bbox": [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                    "confidence": t.confidence,
                    "language": t.language
                }
                for t in self.text_regions
            ],
            "changes": [
                {"type": c.type, "entity_id": c.entity_id, "bbox": [c.bbox.x1, c.bbox.y1, c.bbox.x2, c.bbox.y2] if c.bbox else None, "details": c.details}
                for c in self.changes
            ],
            "processing_time_ms": self.processing_time_ms,
            "inference_mode": self.inference_mode
        }