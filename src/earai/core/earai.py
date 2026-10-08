"""EarAI - Predictive Visual State Orchestrator (3-8 MB)"""
import time
import numpy as np
import torch
import cv2
from typing import Optional, Literal, List, Dict
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

# Import residual encoder functions
from ..heads.residual_encoder import (
    warp_frame, 
    compute_residual_frame, 
    ResidualROI
)

from .config import EarAIConfig, DEFAULT_CONFIG
from .packets import VisionPacket, Entity, BBox, TextRegion, SceneChange

# New components
from ..heads.motion_compensation import MotionEstimator, MotionTransform
from ..heads.adaptive_tokens import TokenLearner, create_adaptive_pooler
from ..heads.state_updater import GatedStateUpdater, UncertaintyEstimator, KeyframeDecider, VisualState
from ..heads.residual_encoder import ResidualEncoder, ROIExtractor, ResidualROI, compute_residual_frame, warp_frame
from ..models.backbone import create_backbone
from ..memory.scene_memory import SceneMemory
from ..heads.foveated_vision import FoveatedVision


@dataclass
class InferenceResult:
    packet: VisionPacket
    backend_time_ms: float
    mode: str
    decision: str = "REUSE"  # REUSE, ROI_CORRECT, KEYFRAME


class EarAI:
    """
    EarAI - Predictive Visual State visual coprocessor.
    
    Architecture:
    - Motion compensation (affine estimation)
    - State warping (predict next state from motion)
    - Residual ROI extraction (only changed/uncertain regions)
    - Adaptive token pooling (TokenLearner-style)
    - GRU-like gated state update
    - Uncertainty estimation
    - Learned keyframe gating (REUSE/ROI_CORRECT/KEYFRAME)
    """
    
    def __init__(self, config: EarAIConfig = None, device: str = "cpu"):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device)
        self.frame_id = 0
        
        # Core components
        self.backbone = create_backbone(self.config).to(self.device).eval()
        
        # Adaptive token pooler (replaces fixed grid)
        self.token_pooler = create_adaptive_pooler({
            "type": "multiscale_tokenlearner",
            "channels_list": [self.config.feature_dim, self.config.feature_dim, self.config.feature_dim, self.config.feature_dim],
            "num_tokens_per_scale": [4, 4, 4, 4],  # 4 tokens per scale = 16 total
            "bottleneck_dim": 64
        }).to(self.device).eval()
        
        # Motion compensation
        self.motion_estimator = MotionEstimator(
            downsample=4,
            max_features=200,
            min_features=10
        )
        
        # Residual encoder
        self.residual_encoder = ResidualEncoder(
            in_channels=9,  # current + warped_prev + diff = 9 channels
            feature_dim=self.config.feature_dim
        ).to(self.device).eval()
        
        # ROI extractor
        self.roi_extractor = ROIExtractor(
            base_resolution=self.config.peripheral_resolution,
            roi_size=(64, 64),
            context_margin=0.1,
            max_rois=8
        )
        
        # State updater
        self.state_updater = GatedStateUpdater(
            state_dim=self.config.feature_dim,
            hidden_dim=512
        ).to(self.device).eval()
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            state_dim=self.config.feature_dim
        ).to(self.device).eval()
        
        # Keyframe decider
        self.keyframe_decider = KeyframeDecider(
            state_dim=self.config.feature_dim,
            reuse_threshold=0.3,
            roi_threshold=0.5,
            max_roi_ratio=0.3
        ).to(self.device).eval()
        
        # Scene memory (persistent entity tracking)
        self.scene_memory = SceneMemory(self.config)
        
        # Foveated vision
        self.foveated_vision = FoveatedVision(self.config)
        
        # Previous frame/state for streaming
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_state: Optional[VisualState] = None
        self.prev_frame_warped: Optional[np.ndarray] = None
        self.image_labels: list[dict] = []  # Scene-level labels, never object detections
        
        # Preprocessing
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # Class names
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
        
        print(f"EarAI initialized on {device}")
        self._print_model_size()
    
    def _print_model_size(self):
        total_params = sum(p.numel() for p in self.backbone.parameters())
        total_params += sum(p.numel() for p in self.token_pooler.parameters())
        total_params += sum(p.numel() for p in self.residual_encoder.parameters())
        total_params += sum(p.numel() for p in self.state_updater.parameters())
        total_params += sum(p.numel() for p in self.uncertainty_estimator.parameters())
        total_params += sum(p.numel() for p in self.keyframe_decider.parameters())
        size_mb = total_params * 1 / 1e6
        print(f"Total params: {total_params/1e6:.2f}M (~{size_mb:.1f} MB INT8)")
    
    def preprocess(self, frame: np.ndarray, resolution: tuple) -> torch.Tensor:
        img = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = (img - self.imagenet_mean) / self.imagenet_std
        return img
    
    def process_frame(self, frame: np.ndarray, force_full: bool = False) -> InferenceResult:
        """
        Main entry point - Predictive Visual State processing.
        
        Pipeline:
        1. Motion estimation (frame t-1 -> t)
        2. State warping (predict H_t from H_{t-1})
        3. Residual computation (I_t - warp(I_{t-1}))
        4. ROI extraction (changes, uncertainty, fovea, text)
        5. Residual encoding (only ROIs)
        6. State update (GRU-style gated correction)
        7. Uncertainty estimation
        8. Keyframe decision (REUSE/ROI_CORRECT/KEYFRAME)
        9. Scene memory update (entity tracking)
        """
        start_total = time.time()
        self.frame_id += 1
        
        # First frame: full keyframe initialization
        if self.frame_id == 1 or force_full:
            return self._keyframe_inference(frame)
        
        # 1. Motion estimation
        motion = self.motion_estimator.estimate(frame)
        if motion is None:
            # Fallback: treat as keyframe
            return self._keyframe_inference(frame)
        
        # 3. Compute residual with valid mask
        if self.prev_frame is not None and motion is not None:
            self.prev_frame_warped, valid_mask = warp_frame(self.prev_frame, motion)
        else:
            self.prev_frame_warped = frame.copy()
            valid_mask = np.ones(frame.shape[:2], dtype=np.uint8)
        
        # 3. Compute residual with valid mask
        residual_map, binary_mask = compute_residual_frame(frame, self.prev_frame_warped, valid_mask)
        residual_magnitude = float(residual_map.mean())
        
        # 4. Extract ROIs
        rois = self.roi_extractor.extract_rois(
            frame=frame,
            motion_transform=motion,
            prev_frame=self.prev_frame,
            change_map=binary_mask,  # Use binary mask for contour detection
            uncertainty_map=None,  # Will be filled after state update
            fovea_requests=[],
            text_regions=[]
        )
        
        # 5. Estimate ROI area ratio for keyframe decision
        roi_area_ratio = sum(
            (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) 
            for r in rois
        )
        
        # 6. Quick uncertainty check from previous state
        prev_uncertainty = 0.0
        prev_unc = 0.0
        if self.prev_state is not None:
            with torch.no_grad():
                prev_tokens = self.prev_state.scene_tokens.unsqueeze(0).to(self.device)
                prev_unc = self.uncertainty_estimator(prev_tokens).max().item()
                prev_uncertainty = prev_unc
        
        # 7. Keyframe decision - use threshold-based (learned gate is untrained)
        decision = self.keyframe_decider.decide_threshold(
            uncertainties=torch.tensor([prev_unc]),
            residual_magnitude=residual_magnitude,
            roi_area_ratio=roi_area_ratio
        )
        
        if decision == "KEYFRAME":
            result = self._keyframe_inference(frame)
            result.decision = "KEYFRAME"
            return result
        elif decision == "REUSE":
            # Reuse previous state with motion compensation
            result = self._reuse_inference(frame, motion)
            result.decision = "REUSE"
            return result
        else:  # ROI_CORRECT
            result = self._roi_correction_inference(frame, motion, rois, residual_magnitude)
            result.decision = "ROI_CORRECT"
            return result
    
    def _keyframe_inference(self, frame: np.ndarray) -> InferenceResult:
        """Full keyframe inference - initialize state from scratch using multi-scale features"""
        start = time.time()
        
        # Preprocess
        peripheral_tensor = self.preprocess(frame, self.config.peripheral_resolution)
        
        # Backbone multi-scale features
        with torch.no_grad():
            backbone_out = self.backbone(peripheral_tensor)
            
            # Get multi-scale features for adaptive pooling
            features = []
            for scale in ['F4', 'F8', 'F16', 'F32']:
                if scale in backbone_out:
                    features.append(backbone_out[scale])
            
            # Multi-scale adaptive token pooling
            if len(features) > 1:
                # Use multi-scale token pooler with attention
                pooler_out = self.token_pooler(features, return_attention=True)
                scene_tokens = pooler_out["tokens"]
                # Store attention info for spatial routing
                self._last_attention = pooler_out.get("attention_per_scale", None)
            else:
                pooler_out = self.token_pooler(features[0], return_attention=True)
                scene_tokens = pooler_out["tokens"]
                self._last_attention = pooler_out.get("attention_per_scale", None)
            
            global_embedding = backbone_out['global']  # [1, D]
            imagenet_logits = backbone_out.get('imagenet_logits')
            
            # ImageNet is a whole-image classifier, NOT a bounding-box detector.
            # Do not fabricate object entities or arbitrary bounding boxes.
            entities = []
            self.image_labels = (
                self._classify_imagenet(imagenet_logits)
                if imagenet_logits is not None else []
            )
            
            # OCR
            text_regions = self._process_ocr_tesseract(frame)
            
            # Fovea requests
            fovea_requests = []
            # TODO: generate from entities
        
        # Extract centroids from attention for spatial routing (concatenate all scales)
        token_centroids_list = []
        if hasattr(self, '_last_attention') and self._last_attention:
            for attn_info in self._last_attention:
                if attn_info and "centroids" in attn_info:
                    cents = attn_info["centroids"].cpu().numpy().squeeze(0)
                    token_centroids_list.append(cents)
        
        if token_centroids_list:
            token_centroids = np.concatenate(token_centroids_list, axis=0)
        else:
            token_centroids = np.zeros((scene_tokens.shape[1], 2), dtype=np.float32)
        
        # Create initial visual state
        state = VisualState(
            scene_tokens=torch.from_numpy(scene_tokens.cpu().numpy().squeeze(0)),
            token_centroids=torch.from_numpy(token_centroids),
            text_regions=text_regions,
            region_tokens=torch.zeros((0, self.config.feature_dim)),
            region_bboxes=torch.zeros((0, 4)),
            entity_tracks={},
            uncertainty=torch.zeros(scene_tokens.shape[1]),
            frame_id=self.frame_id,
            timestamp=time.time()
        )
        
        # Update scene memory
        packet = VisionPacket(
            t=time.time(),
            frame_id=self.frame_id,
            scene_embedding=global_embedding.cpu().numpy().squeeze(),
            entities=entities,
            text_regions=text_regions,
            changes=[],
            fovea_requests=fovea_requests,
            image_labels=list(self.image_labels),
            processing_time_ms=(time.time() - start) * 1000,
            inference_mode="keyframe",
            peripheral_resolution=self.config.peripheral_resolution
        )
        
        packet = self.scene_memory.update(packet)
        state.entity_tracks = {e.id: e for e in packet.entities}
        
        # Store for next frame
        self.prev_frame = frame.copy()
        self.prev_state = state
        
        # Initialize motion estimator with first frame
        self.motion_estimator.estimate(frame)
        
        processing_time = (time.time() - start) * 1000
        
        return InferenceResult(
            packet=packet,
            backend_time_ms=processing_time,
            mode="keyframe",
            decision="KEYFRAME"
        )
    
    def _roi_correction_inference(self, frame: np.ndarray, motion, rois: List[ResidualROI], 
                                   residual_magnitude: float) -> InferenceResult:
        """ROI correction: encode residuals ONLY in changed regions, update state.
        NO full backbone pass - only residual encoder + state update."""
        start = time.time()
        
        # 1. Warp previous frame and compute residual with valid mask
        if self.prev_frame is not None and motion is not None:
            frame_warped, valid_mask = warp_frame(self.prev_frame, motion)
            residual_map, binary_mask = compute_residual_frame(frame, frame_warped, valid_mask)
        else:
            frame_warped = frame.copy()
            valid_mask = np.ones(frame.shape[:2], dtype=np.uint8)
            residual_map, binary_mask = compute_residual_frame(frame, frame_warped, valid_mask)
        
        # 2. Extract and encode residual patches (ONLY this, no full backbone)
        residual_patches = self.roi_extractor.crop_rois(frame, frame_warped, rois)
        if len(residual_patches) > 0:
            residual_patches = residual_patches.unsqueeze(0).to(self.device)  # [1, N, 9, H, W]
            with torch.no_grad():
                delta_features = self.residual_encoder(residual_patches)  # [1, N, D]
        else:
            delta_features = torch.zeros((1, 0, self.config.feature_dim), device=self.device)
        
        # 3. State update using previous state + delta features
        if self.prev_state is not None:
            prev_tokens = self.prev_state.scene_tokens.unsqueeze(0).to(self.device)
            prev_centroids_np = self.prev_state.token_centroids.cpu().numpy()
            warped_centroids_np = np.clip(motion.warp_points(prev_centroids_np), 0.0, 1.0)
            predicted_centroids = torch.as_tensor(
                warped_centroids_np, device=self.device, dtype=prev_tokens.dtype
            ).unsqueeze(0)

            # Route ROI updates at motion-compensated locations, not stale coordinates.
            with torch.no_grad():
                updated_tokens = self.state_updater(
                    predicted_state=prev_tokens,
                    predicted_centroids=predicted_centroids,
                    delta_features=delta_features,
                    delta_bboxes=torch.tensor(np.array([r.bbox for r in rois]), device=self.device).unsqueeze(0) if rois else torch.zeros((1, 0, 4), device=self.device),
                    state_centroids=predicted_centroids
                )
        else:
            # No previous state - this shouldn't happen in ROI_CORRECT, fallback to keyframe
            return self._keyframe_inference(frame)
        
        # 4. Uncertainty estimation
        with torch.no_grad():
            uncertainties = self.uncertainty_estimator(updated_tokens).cpu().numpy().squeeze()
        
        # 5. Create entities from updated tokens
        entities = self._tokens_to_entities(updated_tokens, uncertainties)
        
        # 6. OCR (only on ROIs if needed, for now full frame)
        text_regions = self._process_ocr_tesseract(frame)
        
        # 7. Scene embedding from updated tokens (mean pooling)
        scene_embedding = updated_tokens.cpu().numpy().squeeze(0).mean(axis=0)
        
        # 8. Scene memory update
        packet = VisionPacket(
            t=time.time(),
            frame_id=self.frame_id,
            scene_embedding=scene_embedding,
            entities=entities,
            text_regions=text_regions,
            changes=[],
            fovea_requests=[],
            image_labels=list(self.image_labels),
            processing_time_ms=0,
            inference_mode="roi_correction",
            peripheral_resolution=self.config.peripheral_resolution
        )
        
        packet = self.scene_memory.update(packet)
        
        # 9. Persist motion-compensated token positions.
        updated_centroids = warped_centroids_np.astype(np.float32)

        # Fresh OCR boxes are already in the current frame; never warp them again.
        if text_regions:
            updated_text_regions = text_regions
        else:
            updated_text_regions = [
                TextRegion(
                    value=tr.value,
                    bbox=BBox(*motion.warp_bbox(np.array([
                        tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2
                    ]))),
                    confidence=tr.confidence,
                    language=tr.language,
                )
                for tr in self.prev_state.text_regions
            ]

        # 10. Update state
        new_state = VisualState(
            scene_tokens=torch.from_numpy(updated_tokens.cpu().numpy().squeeze(0)),
            token_centroids=torch.from_numpy(updated_centroids),
            text_regions=updated_text_regions,
            region_tokens=torch.zeros((0, self.config.feature_dim)),
            region_bboxes=torch.from_numpy(np.array([r.bbox for r in rois]) if rois else np.zeros((0, 4))),
            entity_tracks={e.id: e for e in packet.entities},
            uncertainty=torch.from_numpy(uncertainties),
            frame_id=self.frame_id,
            timestamp=time.time()
        )
        
        # Store for next frame
        self.prev_frame = frame.copy()
        self.prev_state = new_state
        
        processing_time = (time.time() - start) * 1000
        packet.processing_time_ms = processing_time
        
        return InferenceResult(
            packet=packet,
            backend_time_ms=processing_time,
            mode="roi_correction",
            decision="ROI_CORRECT"
        )
    def _reuse_inference(self, frame: np.ndarray, motion) -> InferenceResult:
            """Fast path: reuse previous state with motion compensation"""
            start = time.time()
        
            # Warp visual state (tokens + centroids + text_regions) with motion
            if self.prev_state is not None:
                # Warp token centroids
                prev_centroids = self.prev_state.token_centroids.clone()
                warped_np = np.clip(
                    motion.warp_points(prev_centroids.cpu().numpy()), 0.0, 1.0
                )
                warped_centroids = torch.as_tensor(
                    warped_np, dtype=prev_centroids.dtype
                )
            
                # Warp text regions
                warped_text_regions = []
                for tr in self.prev_state.text_regions:
                    warped_bbox = motion.warp_bbox(np.array([tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2]))
                    warped_text_regions.append(TextRegion(
                        value=tr.value,
                        bbox=BBox(*warped_bbox),
                        confidence=tr.confidence,
                        language=tr.language
                    ))
            
                # Update entity positions with motion
                entities = []
                for entity in self.scene_memory.entities.values():
                    e = entity.entity
                    warped_bbox = motion.warp_bbox(np.array([e.bbox.x1, e.bbox.y1, e.bbox.x2, e.bbox.y2]))
                    e.bbox = BBox(*warped_bbox)
                    entities.append(e)
            
                # Update state with warped centroids and text_regions (semantic tokens unchanged)
                self.prev_state.token_centroids = warped_centroids
                self.prev_state.text_regions = warped_text_regions
                text_regions = warped_text_regions
            else:
                entities = []
                text_regions = []
        
            packet = VisionPacket(
                t=time.time(),
                frame_id=self.frame_id,
                scene_embedding=self.prev_state.scene_tokens.mean(axis=0) if self.prev_state else np.zeros(self.config.feature_dim),
                entities=entities,
                text_regions=text_regions,
                changes=[],
                fovea_requests=[],
                image_labels=list(self.image_labels),
                processing_time_ms=(time.time() - start) * 1000,
                inference_mode="reuse",
                peripheral_resolution=self.config.peripheral_resolution
            )
        
            self.prev_frame = frame.copy()
            # State already updated above
        
            return InferenceResult(
                packet=packet,
                backend_time_ms=packet.processing_time_ms,
                mode="reuse",
                decision="REUSE"
            )
    
    def _tokens_to_entities(self, tokens: torch.Tensor, uncertainties: np.ndarray) -> List[Entity]:
        """No detector head has been trained to output localized entities yet.

        Visual tokens/uncertainty alone do NOT establish object boxes or labels.
        Returning empty detections is safer than inventing a constant box.
        """
        return []

    @torch.no_grad()
    def _classify_imagenet(self, imagenet_logits: torch.Tensor) -> list[dict]:
        """Image-wide ImageNet predictions, not object detections."""
        probabilities = torch.softmax(imagenet_logits.reshape(-1), dim=0)
        top = torch.topk(probabilities, min(5, probabilities.numel()))
        labels = []
        for index, confidence in zip(top.indices.tolist(), top.values.tolist()):
            if confidence < 0.1:
                continue
            class_name = (
                IMAGENET_CATEGORIES[index]
                if IMAGENET_CATEGORIES and index < len(IMAGENET_CATEGORIES)
                else f"imagenet_class_{index}"
            )
            labels.append({
                "class": class_name,
                "confidence": float(confidence),
                "source_frame_id": self.frame_id,
                "scope": "whole_image",
            })
        return labels

    def _process_ocr_tesseract(self, frame: np.ndarray) -> List[TextRegion]:
        text_regions = []
        if not TESSERACT_AVAILABLE:
            return text_regions
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lang = getattr(self, "_ocr_lang", "eng+rus")
            try:
                data = pytesseract.image_to_data(
                    rgb, output_type=pytesseract.Output.DICT, lang=lang
                )
            except pytesseract.TesseractError:
                if lang == "eng":
                    raise
                self._ocr_lang = "eng"  # Russian language pack may not be installed
                data = pytesseract.image_to_data(
                    rgb, output_type=pytesseract.Output.DICT, lang="eng"
                )
            n_boxes = len(data['text'])
            h, w = frame.shape[:2]
            for i in range(n_boxes):
                text = data['text'][i].strip()
                try:
                    conf = float(data["conf"][i])
                except (ValueError, TypeError):
                    continue
                if text and conf > 30:
                    x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = BBox(x / w, y / h, (x + bw) / w, (y + bh) / h)
                    text_regions.append(TextRegion(value=text, bbox=bbox, confidence=conf / 100.0, language='mixed'))
        except Exception as e:
            print(f"OCR error: {e}")
        return text_regions
    
    def get_universal_api(self, packet: VisionPacket) -> dict:
        return packet.to_universal_api()
    
    def get_native_api(self, packet: VisionPacket) -> dict:
        native = packet.to_native_api()
        native["world_state"] = self.scene_memory.get_world_state()
        if hasattr(packet, "fovea_crops"):
            native["fovea_crops"] = {k: v.tolist() for k, v in packet.fovea_crops.items()}
        return native
    
    def request_fovea(self, entity_id: int, zoom: float = 3.0) -> Optional[dict]:
        return self.scene_memory.request_fovea(entity_id, zoom)
    
    def reset(self):
        self.motion_estimator.reset()
        self.scene_memory = SceneMemory(self.config)
        self.prev_frame = None
        self.prev_state = None
        self.prev_frame_warped = None
        self.image_labels = []
        self.frame_id = 0


def create_earai(config: EarAIConfig = None, device: str = "cpu") -> EarAI:
    return EarAI(config, device)