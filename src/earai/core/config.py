"""EarAI configuration - target 3-8 MB total"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class EarAIConfig:
    # Model size targets
    target_params_mb: float = 4.0  # 3-5 MB INT8
    quantize: Literal["int8", "int4", "fp16", "fp32"] = "int8"

    # Backbone (MobileNetV3-Small ~2.5M params)
    backbone: str = "mobilenetv3_small"
    backbone_pretrained: bool = True
    backbone_width_mult: float = 1.0
    feature_dim: int = 256  # Common dimension for all heads

    # Heads (lightweight)
    object_head_dim: int = 128
    semantic_head_dim: int = 256
    motion_head_dim: int = 64
    ocr_head_dim: int = 128
    change_head_dim: int = 32

    # Scene memory
    max_entities: int = 100
    entity_embedding_dim: int = 256
    num_scene_tokens: int = 16  # Adaptive tokens (8-16), not max_entities
    memory_decay: float = 0.99
    association_iou_threshold: float = 0.3
    association_visual_threshold: float = 0.7

    # Foveated vision
    peripheral_resolution: tuple = (256, 256)
    fovea_resolution: tuple = (600, 600)
    fovea_trigger_threshold: float = 0.5

    # Streaming
    change_detection_threshold: float = 0.02
    full_inference_interval: int = 3  # Run full NN every N frames
    max_fps: int = 30

    # I/O
    input_sources: list = None  # ["camera", "screen", "file"]

    def __post_init__(self):
        if self.input_sources is None:
            self.input_sources = ["camera"]


DEFAULT_CONFIG = EarAIConfig()