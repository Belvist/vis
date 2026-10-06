"""API Layer - Universal (JSON) and Native (embeddings) interfaces"""
import json
import time
import cv2
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from ..core.earai import EarAI, InferenceResult
from ..core.config import EarAIConfig
from ..core.packets import VisionPacket


@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: dict
    error: Optional[str] = None
    latency_ms: float = 0.0


class UniversalAPI:
    """
    Universal JSON/text API - works with ANY LLM (GPT, Claude, local, etc.)
    Output: Human-readable scene descriptions + structured data
    """

    def __init__(self, earai: EarAI):
        self.earai = earai

    def process_frame(self, frame, force_full: bool = False) -> APIResponse:
        """Process frame and return universal format"""
        start = time.time()
        try:
            result: InferenceResult = self.earai.process_frame(frame, force_full)
            data = self.earai.get_universal_api(result.packet)
            data["meta"] = {
                "frame_id": result.packet.frame_id,
                "inference_mode": result.mode,
                "decision": result.decision,
                "processing_time_ms": round(result.packet.processing_time_ms, 1),
                "timestamp": result.packet.t
            }
            return APIResponse(
                success=True,
                data=data,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    def get_scene_description(self, packet: VisionPacket) -> str:
        """Generate natural language scene description"""
        parts = []

        if packet.entities:
            entity_desc = []
            for e in packet.entities[:5]:
                pos = "left" if e.bbox.center()[0] < 0.33 else "center" if e.bbox.center()[0] < 0.66 else "right"
                dist = "near" if e.bbox.area() > 0.1 else "far"
                entity_desc.append(f"{e.class_name} ({pos}, {dist})")
            parts.append(f"Objects: {', '.join(entity_desc)}")

        if packet.text_regions:
            texts = [f'"{t.value}"' for t in packet.text_regions[:3]]
            parts.append(f"Text: {', '.join(texts)}")

        if packet.changes:
            changes = [f"{c.type} ({c.details})" for c in packet.changes]
            parts.append(f"Changes: {', '.join(changes)}")

        if packet.fovea_requests:
            parts.append(f"Fovea requested for {len(packet.fovea_requests)} regions")

        return ". ".join(parts) if parts else "Empty scene."

    def to_prompt_context(self, packet: VisionPacket, max_tokens: int = 500) -> str:
        """Format as compact prompt context for LLM"""
        desc = self.get_scene_description(packet)
        return f"[VISION t={packet.t:.3f}] {desc}"


class NativeAPI:
    """
    Native API - rich embeddings + full state for trained models/agents
    Output: Raw embeddings, entity vectors, spatial/temporal state
    """

    def __init__(self, earai: EarAI):
        self.earai = earai

    def process_frame(self, frame, force_full: bool = False) -> APIResponse:
        """Process frame and return native format"""
        start = time.time()
        try:
            result: InferenceResult = self.earai.process_frame(frame, force_full)
            data = self.earai.get_native_api(result.packet)
            data["meta"] = {
                "frame_id": result.packet.frame_id,
                "inference_mode": result.mode,
                "decision": result.decision,
                "processing_time_ms": round(result.packet.processing_time_ms, 1),
                "timestamp": result.packet.t
            }
            return APIResponse(
                success=True,
                data=data,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    def get_entity_embeddings(self, packet: VisionPacket) -> dict:
        """Extract entity embeddings for downstream model"""
        return {
            "entity_ids": [e.id for e in packet.entities],
            "visual_embeddings": [e.visual_embedding.tolist() for e in packet.entities],
            "semantic_embeddings": [e.semantic_embedding.tolist() for e in packet.entities],
            "motion_vectors": [e.motion_vector.tolist() for e in packet.entities],
            "bboxes": [[e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2] for e in packet.entities],
            "classes": [e.class_name for e in packet.entities],
            "confidences": [e.confidence for e in packet.entities]
        }

    def get_scene_embedding(self, packet: VisionPacket) -> np.ndarray:
        """Get global scene embedding"""
        return packet.scene_embedding.copy()

    def get_temporal_state(self) -> dict:
        """Get temporal/world state from memory"""
        return self.earai.scene_memory.get_world_state()


class StreamingAPI:
    """
    High-level streaming interface for continuous operation
    Handles: camera, screen, file sources
    """

    def __init__(self, config: EarAIConfig = None, device: str = "cpu",
                 on_frame: Optional[Callable[[VisionPacket], None]] = None,
                 api_mode: str = "universal"):
        self.earai = EarAI(config, device)
        self.on_frame = on_frame
        self.api_mode = api_mode
        self.running = False
        self.frame_callback = None

        if api_mode == "universal":
            self.api = UniversalAPI(self.earai)
        else:
            self.api = NativeAPI(self.earai)

    def start_camera(self, camera_id: int = 0, fps: int = 30):
        """Start processing camera stream"""
        import threading
        self.running = True

        def camera_loop():
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FPS, fps)
            frame_interval = 1.0 / fps

            while self.running:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.api.process_frame(frame)
                if result.success and self.on_frame:
                    if self.api_mode == "universal":
                        self.on_frame(result.data)
                    else:
                        self.on_frame(result.data)

                # Maintain FPS
                elapsed = time.time() - start
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)

            cap.release()

        self.thread = threading.Thread(target=camera_loop, daemon=True)
        self.thread.start()

    def start_screen(self, monitor: int = 0, fps: int = 10):
        """Start screen capture (requires mss)"""
        import threading
        try:
            import mss
        except ImportError:
            raise RuntimeError("mss required for screen capture: pip install mss")

        self.running = True

        def screen_loop():
            with mss.mss() as sct:
                monitor_info = sct.monitors[monitor + 1]
                frame_interval = 1.0 / fps

                while self.running:
                    start = time.time()
                    screenshot = sct.grab(monitor_info)
                    frame = np.array(screenshot)[:, :, :3]  # BGRA -> BGR

                    result = self.api.process_frame(frame)
                    if result.success and self.on_frame:
                        if self.api_mode == "universal":
                            self.on_frame(result.data)
                        else:
                            self.on_frame(result.data)

                    elapsed = time.time() - start
                    sleep_time = max(0, frame_interval - elapsed)
                    time.sleep(sleep_time)

        self.thread = threading.Thread(target=screen_loop, daemon=True)
        self.thread.start()

    def process_image(self, image_path: str) -> APIResponse:
        """Process single image file"""
        frame = cv2.imread(image_path)
        if frame is None:
            return APIResponse(success=False, data={}, error=f"Cannot read {image_path}")
        return self.api.process_frame(frame, force_full=True)

    def process_video(self, video_path: str, callback=None, every_n: int = 1):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                result = self.api.process_frame(frame, force_full=(frame_idx == 0))
                if result.success:
                    if callback:
                        callback(result.data, frame_idx)
                    elif self.on_frame:
                        self.on_frame(result.data)

            frame_idx += 1

        cap.release()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)

    def request_fovea(self, entity_id: int, zoom: float = 3.0):
        """Request high-res crop"""
        return self.earai.request_fovea(entity_id, zoom)

    def reset(self):
        self.earai.reset()


# Convenience function
def create_streaming_api(config: EarAIConfig = None, device: str = "cpu",
                        api_mode: str = "universal",
                        on_frame: Optional[Callable] = None) -> StreamingAPI:
    return StreamingAPI(config, device, on_frame, api_mode)