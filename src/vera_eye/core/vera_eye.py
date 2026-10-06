"""Vera Eye - Main orchestrator"""
import time
import numpy as np
import torch
import cv2
from typing import Optional, Literal
from dataclasses import dataclass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Load ImageNet categories from torchvision
try:
    import torchvision
    IMAGENET_CATEGORIES = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta['categories']
except Exception:
    IMAGENET_CATEGORIES = None

from .config import VeraEyeConfig, DEFAULT_CONFIG
from .packets import VisionPacket, Entity, BBox, TextRegion, SceneChange
from ..heads.change_detector import ChangeDetector
from ..models.backbone import TinyBackbone, create_backbone
from ..heads.heads import create_heads
from ..memory.scene_memory import SceneMemory
from ..heads.foveated_vision import FoveatedVision


@dataclass
class InferenceResult:
    packet: VisionPacket
    backend_time_ms: float
    mode: str


class VeraEye:
    """
    Main Vera Eye system - 3-8 MB visual coprocessor
    """

    def __init__(self, config: VeraEyeConfig = None, device: str = "cpu"):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device)
        self.frame_id = 0
        self.last_full_inference = 0

        # Components
        self.change_detector = ChangeDetector(self.config)
        self.backbone = create_backbone(self.config).to(self.device).eval()
        self.heads = create_heads(self.config)
        for name, head in self.heads.items():
            head.to(self.device).eval()
        self.scene_memory = SceneMemory(self.config)
        self.foveated_vision = FoveatedVision(self.config)

        # Preprocessing
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # Class names (COCO subset + custom)
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush", "screen", "text", "button", "sign", "door", "window", "hand"
        ]

        print(f"Vera Eye initialized on {device}")
        self._print_model_size()

    def _print_model_size(self):
        total_params = sum(p.numel() for p in self.backbone.parameters())
        for head in self.heads.values():
            total_params += sum(p.numel() for p in head.parameters())
        size_mb = total_params * 1 / 1e6  # INT8 ~1 byte/param
        print(f"Total params: {total_params/1e6:.2f}M (~{size_mb:.1f} MB INT8)")

    def preprocess(self, frame: np.ndarray, resolution: tuple) -> torch.Tensor:
        """Preprocess frame for backbone"""
        # Resize
        img = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = (img - self.imagenet_mean) / self.imagenet_std
        return img

    def run_full_inference(self, frame: np.ndarray) -> VisionPacket:
        """Full neural inference - runs every N frames or on significant change"""
        start = time.time()

        # Preprocess
        peripheral_tensor = self.preprocess(frame, self.config.peripheral_resolution)

        # Backbone
        with torch.no_grad():
            backbone_out = self.backbone(peripheral_tensor)
            global_emb = backbone_out["global_embedding"]  # [1, D]
            spatial_feat = backbone_out["spatial_features"]  # [1, 576, 7, 7]
            imagenet_logits = backbone_out.get("imagenet_logits", None)

            # Heads (Vera Eye custom heads - for future distillation)
            obj_out = self.heads["object_region"](spatial_feat)
            sem_out = self.heads["semantic"](global_emb)

            # Process detections from custom heads (placeholder until distillation)
            entities = self._process_detections(obj_out, sem_out, global_emb, spatial_feat)

            # ImageNet classification (works NOW with pretrained weights)
            if imagenet_logits is not None:
                imagenet_entities = self._process_imagenet(imagenet_logits)
                entities.extend(imagenet_entities)

            # OCR with Tesseract (works NOW)
            text_regions = self._process_ocr_tesseract(frame)

            # Scene embedding
            scene_embedding = sem_out["scene_embedding"].cpu().numpy().squeeze()

        processing_time = (time.time() - start) * 1000

        packet = VisionPacket(
            t=time.time(),
            frame_id=self.frame_id,
            scene_embedding=scene_embedding,
            entities=entities,
            text_regions=text_regions,
            changes=[],  # Filled by scene memory
            fovea_requests=[],
            processing_time_ms=processing_time,
            inference_mode="full",
            peripheral_resolution=self.config.peripheral_resolution
        )

        return packet

    def run_tracking_only(self, frame: np.ndarray) -> VisionPacket:
        """Lightweight tracking-only inference - runs on intermediate frames"""
        start = time.time()

        # Use change detector + scene memory prediction
        change_result = self.change_detector.detect(frame)

        # Quick peripheral embedding only
        peripheral_tensor = self.preprocess(frame, self.config.peripheral_resolution)
        with torch.no_grad():
            backbone_out = self.backbone(peripheral_tensor)
            global_emb = backbone_out["global_embedding"]
            scene_embedding = global_emb.cpu().numpy().squeeze()

        # Return predicted entities from memory
        entities = [e.entity for e in self.scene_memory.entities.values()
                   if e.confirmed or e.entity.frames_tracked > 1]

        processing_time = (time.time() - start) * 1000

        packet = VisionPacket(
            t=time.time(),
            frame_id=self.frame_id,
            scene_embedding=scene_embedding,
            entities=entities,
            text_regions=[],  # Skip OCR in tracking mode
            changes=[],
            fovea_requests=[],
            processing_time_ms=processing_time,
            inference_mode="tracking_only",
            peripheral_resolution=self.config.peripheral_resolution
        )

        return packet

    def process_frame(self, frame: np.ndarray, force_full: bool = False) -> InferenceResult:
        """Main entry point - processes one frame"""
        self.frame_id += 1

        # 1. Change detection (every frame, ultra-fast)
        change_result = self.change_detector.detect(frame)

        # 2. Decide inference mode
        needs_full = (
            force_full or
            self.frame_id == 1 or
            change_result["significant_change"] or
            (self.frame_id - self.last_full_inference) >= self.config.full_inference_interval
        )

        if needs_full:
            packet = self.run_full_inference(frame)
            self.last_full_inference = self.frame_id
            mode = "full"
        else:
            packet = self.run_tracking_only(frame)
            mode = "tracking_only"

        # 3. Update scene memory (persistent world state)
        packet = self.scene_memory.update(packet)

        # 4. Auto-generate fovea requests
        fovea_requests = self.foveated_vision.auto_fovea_requests(packet)
        packet.fovea_requests = fovea_requests
        for req in fovea_requests:
            self.foveated_vision.add_fovea_request(req)

        # 5. Extract fovea crops if requested
        if fovea_requests:
            packet = self.foveated_vision.process_fovea_requests(frame, packet)

        return InferenceResult(packet=packet, backend_time_ms=packet.processing_time_ms, mode=mode)

    def _process_detections(self, obj_out: dict, sem_out: dict,
                           global_emb: torch.Tensor, spatial_feat: torch.Tensor) -> list[Entity]:
        """Convert head outputs to Entity objects"""
        entities = []

        objectness = obj_out["objectness"].squeeze()  # [7, 7]
        bboxes = obj_out["bboxes"].squeeze().permute(1, 2, 0)  # [7, 7, 4]
        embeddings = obj_out["embeddings"].squeeze().permute(1, 2, 0)  # [7, 7, D]

        class_logits = sem_out["class_logits"].squeeze()  # [num_classes]
        class_probs = torch.softmax(class_logits, dim=-1)

        # Threshold objectness
        threshold = 0.3
        h, w = objectness.shape
        for i in range(h):
            for j in range(w):
                if objectness[i, j] > threshold:
                    # Get bbox (already normalized 0-1)
                    bbox_data = bboxes[i, j].cpu().numpy()
                    bbox = BBox(
                        float(np.clip(bbox_data[0], 0, 1)),
                        float(np.clip(bbox_data[1], 0, 1)),
                        float(np.clip(bbox_data[2], 0, 1)),
                        float(np.clip(bbox_data[3], 0, 1))
                    )

                    # Visual embedding
                    vis_emb = embeddings[i, j].cpu().numpy()
                    vis_emb = vis_emb / (np.linalg.norm(vis_emb) + 1e-6)

                    # Semantic embedding (from global)
                    sem_emb = sem_out["scene_embedding"].squeeze().cpu().numpy()

                    # Class prediction
                    class_idx = int(torch.argmax(class_probs).item())
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
                    confidence = float(class_probs[class_idx].item()) * float(objectness[i, j].item())

                    # Motion (will be updated by tracker)
                    motion = np.array([0.0, 0.0], dtype=np.float32)

                    entity = Entity(
                        id=-1,  # Will be assigned by scene memory
                        bbox=bbox,
                        visual_embedding=vis_emb,
                        semantic_embedding=sem_emb,
                        motion_vector=motion,
                        class_name=class_name,
                        confidence=confidence,
                        first_seen=time.time(),
                        last_seen=time.time(),
                        frames_tracked=1
                    )
                    entities.append(entity)

        # NMS - keep top detections
        if len(entities) > 20:
            entities.sort(key=lambda e: e.confidence, reverse=True)
            entities = entities[:20]

        return entities

    def _process_imagenet(self, imagenet_logits: torch.Tensor) -> list[Entity]:
        """Process ImageNet classification logits to entities"""
        entities = []
        probs = torch.softmax(imagenet_logits.squeeze(), dim=-1)
        top5 = torch.topk(probs, 5)
        
        for idx, conf in zip(top5.indices.tolist(), top5.values.tolist()):
            if conf > 0.1:  # Threshold
                if IMAGENET_CATEGORIES and idx < len(IMAGENET_CATEGORIES):
                    class_name = IMAGENET_CATEGORIES[idx]
                else:
                    class_name = f"class_{idx}"
                # Create a central bbox for the classified object
                bbox = BBox(0.25, 0.25, 0.75, 0.75)
                entity = Entity(
                    id=-1,
                    bbox=bbox,
                    visual_embedding=np.zeros(256, dtype=np.float32),
                    semantic_embedding=np.zeros(256, dtype=np.float32),
                    motion_vector=np.array([0.0, 0.0], dtype=np.float32),
                    class_name=class_name,
                    confidence=float(conf),
                    first_seen=time.time(),
                    last_seen=time.time(),
                    frames_tracked=1
                )
                entities.append(entity)
        return entities

    def _process_ocr_tesseract(self, frame: np.ndarray) -> list[TextRegion]:
        """Process OCR using Tesseract"""
        text_regions = []
        
        if not TESSERACT_AVAILABLE:
            return text_regions
        
        try:
            # Convert to RGB for tesseract
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, lang='eng+rus')
            
            n_boxes = len(data['text'])
            h, w = frame.shape[:2]
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if text and conf > 30:  # Confidence threshold
                    x = data['left'][i]
                    y = data['top'][i]
                    bw = data['width'][i]
                    bh = data['height'][i]
                    
                    # Normalize to 0-1
                    bbox = BBox(
                        x / w, y / h,
                        (x + bw) / w, (y + bh) / h
                    )
                    
                    text_regions.append(TextRegion(
                        value=text,
                        bbox=bbox,
                        confidence=conf / 100.0,
                        language='mixed'
                    ))
        except Exception as e:
            print(f"OCR error: {e}")
        
        return text_regions

    def get_universal_api(self, packet: VisionPacket) -> dict:
        """Get JSON output for any LLM"""
        return packet.to_universal_api()

    def get_native_api(self, packet: VisionPacket) -> dict:
        """Get rich output for native model"""
        native = packet.to_native_api()
        native["world_state"] = self.scene_memory.get_world_state()
        if hasattr(packet, "fovea_crops"):
            native["fovea_crops"] = {
                k: v.tolist() for k, v in packet.fovea_crops.items()
            }
        return native

    def request_fovea(self, entity_id: int, zoom: float = 3.0) -> Optional[dict]:
        """Request high-res crop for entity"""
        return self.scene_memory.request_fovea(entity_id, zoom)

    def reset(self):
        """Reset all state"""
        self.change_detector.reset()
        self.scene_memory = SceneMemory(self.config)
        self.frame_id = 0
        self.last_full_inference = 0


def create_vera_eye(config: VeraEyeConfig = None, device: str = "cpu") -> VeraEye:
    """Factory function"""
    return VeraEye(config, device)