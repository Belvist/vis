"""Persistent Scene Memory - maintains world state across frames"""
import numpy as np
import time
from typing import Optional, List
from dataclasses import dataclass, field
from ..core.packets import Entity, BBox, VisionPacket
from ..core.config import VeraEyeConfig


@dataclass
class TrackedEntity:
    """Internal representation with tracking state"""
    entity: Entity
    predicted_bbox: Optional[BBox] = None
    missed_frames: int = 0
    confirmed: bool = False
    confirmation_frames: int = 3


class SceneMemory:
    """
    Maintains persistent scene state:
    - Entity tracking with IDs over time
    - Visual/semantic memory
    - Spatial relationships
    - Temporal consistency
    """

    def __init__(self, config: VeraEyeConfig):
        self.config = config
        self.entities: dict[int, TrackedEntity] = {}
        self.next_entity_id = 1
        self.frame_count = 0
        self.last_global_embedding: Optional[np.ndarray] = None
        self.last_update_time = time.time()

        # Spatial index for fast association
        self.spatial_grid_size = 10  # 10x10 grid

    def update(self, packet: VisionPacket) -> VisionPacket:
        """Update memory with new detections, return enriched packet"""
        self.frame_count += 1
        current_time = time.time()

        # 1. Predict entity positions (motion model)
        self._predict_positions()

        # 2. Associate detections with existing tracks
        associations = self._associate_detections(packet.entities)

        # 3. Update matched tracks, create new, mark missed
        self._update_tracks(packet.entities, associations, current_time)

        # 4. Remove stale tracks
        self._prune_stale(current_time)

        # 5. Build enriched entity list for output
        output_entities = self._build_output_entities()

        # 6. Detect changes
        changes = self._detect_changes(packet, output_entities)

        # 7. Update global embedding
        self.last_global_embedding = packet.scene_embedding.copy()

        # Create enriched packet
        enriched = VisionPacket(
            t=packet.t,
            frame_id=packet.frame_id,
            scene_embedding=packet.scene_embedding,
            entities=output_entities,
            text_regions=packet.text_regions,
            changes=changes,
            fovea_requests=packet.fovea_requests,
            processing_time_ms=packet.processing_time_ms,
            inference_mode=packet.inference_mode,
            peripheral_resolution=packet.peripheral_resolution
        )

        self.last_update_time = current_time
        return enriched

    def _predict_positions(self):
        """Simple linear motion prediction"""
        for track in self.entities.values():
            if track.entity.frames_tracked > 2:
                # Predict next position from motion vector
                e = track.entity
                cx, cy = e.bbox.center()
                dx, dy = e.motion_vector
                pred_cx = np.clip(cx + dx, 0, 1)
                pred_cy = np.clip(cy + dy, 0, 1)
                w = e.bbox.x2 - e.bbox.x1
                h = e.bbox.y2 - e.bbox.y1
                track.predicted_bbox = BBox(
                    pred_cx - w/2, pred_cy - h/2,
                    pred_cx + w/2, pred_cy + h/2
                )

    def _associate_detections(self, detections: List[Entity]) -> List[tuple]:
        """
        Hungarian-style association: detection -> track
        Returns list of (det_idx, track_id) pairs
        """
        if not self.entities or not detections:
            return []

        # Cost matrix: lower = better match
        track_ids = list(self.entities.keys())
        costs = np.full((len(detections), len(track_ids)), 10.0)

        for i, det in enumerate(detections):
            for j, tid in enumerate(track_ids):
                track = self.entities[tid]

                # IoU cost
                iou = det.bbox.iou(track.entity.bbox)
                if track.predicted_bbox:
                    iou_pred = det.bbox.iou(track.predicted_bbox)
                    iou = max(iou, iou_pred)

                # Visual similarity
                vis_sim = np.dot(det.visual_embedding, track.entity.visual_embedding) / (
                    np.linalg.norm(det.visual_embedding) * np.linalg.norm(track.entity.visual_embedding) + 1e-6
                )

                # Combined cost
                if iou > self.config.association_iou_threshold and vis_sim > self.config.association_visual_threshold:
                    costs[i, j] = (1 - iou) * 0.5 + (1 - vis_sim) * 0.5

        # Simple greedy assignment (good enough for small N)
        associations = []
        used_tracks = set()
        used_dets = set()

        # Sort by cost
        indices = np.argwhere(costs < 5.0)
        if len(indices) > 0:
            sorted_idx = indices[np.argsort(costs[indices[:, 0], indices[:, 1]])]
            for det_idx, track_idx in sorted_idx:
                if det_idx not in used_dets and track_idx not in used_tracks:
                    associations.append((det_idx, track_ids[track_idx]))
                    used_dets.add(det_idx)
                    used_tracks.add(track_idx)

        return associations

    def _update_tracks(self, detections: List[Entity], associations: List[tuple], current_time: float):
        matched_tracks = set()
        matched_dets = set()

        for det_idx, track_id in associations:
            det = detections[det_idx]
            track = self.entities[track_id]

            # Smooth update
            alpha = 0.7
            track.entity.bbox = BBox(
                alpha * track.entity.bbox.x1 + (1-alpha) * det.bbox.x1,
                alpha * track.entity.bbox.y1 + (1-alpha) * det.bbox.y1,
                alpha * track.entity.bbox.x2 + (1-alpha) * det.bbox.x2,
                alpha * track.entity.bbox.y2 + (1-alpha) * det.bbox.y2
            )

            # Update embeddings with EMA
            track.entity.visual_embedding = (
                alpha * track.entity.visual_embedding + (1-alpha) * det.visual_embedding
            )
            track.entity.visual_embedding /= np.linalg.norm(track.entity.visual_embedding) + 1e-6

            track.entity.semantic_embedding = (
                alpha * track.entity.semantic_embedding + (1-alpha) * det.semantic_embedding
            )

            # Motion vector
            cx1, cy1 = track.entity.bbox.center()
            # Use previous center from before update
            prev_cx = cx1  # approximate
            track.entity.motion_vector = det.motion_vector

            track.entity.confidence = max(track.entity.confidence, det.confidence)
            track.entity.last_seen = current_time
            track.entity.frames_tracked += 1
            track.missed_frames = 0

            if not track.confirmed and track.entity.frames_tracked >= track.confirmation_frames:
                track.confirmed = True

            matched_tracks.add(track_id)
            matched_dets.add(det_idx)

        # Unmatched tracks - increment missed
        for tid in self.entities:
            if tid not in matched_tracks:
                self.entities[tid].missed_frames += 1

        # Unmatched detections - create new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det, current_time)

    def _create_track(self, det: Entity, current_time: float):
        if len(self.entities) >= self.config.max_entities:
            # Remove oldest unconfirmed
            candidates = [(tid, t) for tid, t in self.entities.items() if not t.confirmed]
            if candidates:
                oldest = min(candidates, key=lambda x: x[1].entity.first_seen)
                del self.entities[oldest[0]]
            else:
                return  # At capacity

        track = TrackedEntity(
            entity=Entity(
                id=self.next_entity_id,
                bbox=det.bbox,
                visual_embedding=det.visual_embedding.copy(),
                semantic_embedding=det.semantic_embedding.copy(),
                motion_vector=det.motion_vector.copy(),
                class_name=det.class_name,
                confidence=det.confidence,
                first_seen=current_time,
                last_seen=current_time,
                frames_tracked=1
            )
        )
        self.entities[self.next_entity_id] = track
        self.next_entity_id += 1

    def _prune_stale(self, current_time: float):
        to_remove = []
        for tid, track in self.entities.items():
            if track.entity.is_stale(current_time, max_age=3.0):
                to_remove.append(tid)
            elif track.missed_frames > 30 and not track.confirmed:
                to_remove.append(tid)
        for tid in to_remove:
            del self.entities[tid]

    def _build_output_entities(self) -> List[Entity]:
        return [t.entity for t in self.entities.values() if t.confirmed or t.entity.frames_tracked >= 1]

    def _detect_changes(self, packet: VisionPacket, current_entities: List[Entity]) -> List:
        from ..core.packets import SceneChange
        changes = []

        # Current entity IDs
        current_ids = {e.id for e in current_entities}
        prev_ids = set(self.entities.keys()) - current_ids  # Approximate

        # Appeared
        for e in current_entities:
            if e.frames_tracked == 1:
                changes.append(SceneChange("appeared", e.id, e.bbox, f"{e.class_name} appeared"))

        # Disappeared (tracks that were confirmed but now gone)
        for tid in prev_ids:
            if tid in self.entities and self.entities[tid].confirmed:
                changes.append(SceneChange("disappeared", tid, None, f"Entity {tid} disappeared"))

        # Moved
        for e in current_entities:
            if e.frames_tracked > 1 and np.linalg.norm(e.motion_vector) > 0.02:
                changes.append(SceneChange("moved", e.id, e.bbox, f"{e.class_name} moved"))

        return changes

    def get_world_state(self) -> dict:
        """Export full world state for native API"""
        return {
            "entities": [
                {
                    "id": e.id,
                    "class": e.class_name,
                    "bbox": [e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2],
                    "position": e.bbox.center(),
                    "velocity": e.motion_vector.tolist(),
                    "age": time.time() - e.first_seen,
                    "confidence": e.confidence,
                    "holding": e.metadata.get("holding"),
                    "relations": e.metadata.get("relations", [])
                }
                for e in self.entities.values() if e.confirmed
            ],
            "entity_count": len([e for e in self.entities.values() if e.confirmed]),
            "frame": self.frame_count,
            "timestamp": time.time()
        }

    def request_fovea(self, entity_id: int, zoom: float = 3.0) -> Optional[dict]:
        """Generate fovea request for specific entity"""
        if entity_id in self.entities:
            e = self.entities[entity_id].entity
            cx, cy = e.bbox.center()
            return {
                "type": "foveate",
                "entity_id": entity_id,
                "center": [cx, cy],
                "zoom": zoom,
                "priority": "high" if e.confidence > 0.8 else "normal"
            }
        return None