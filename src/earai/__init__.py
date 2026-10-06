"""EarAI - Ultra-lightweight visual coprocessor (3-8 MB)"""

from .core.config import EarAIConfig, DEFAULT_CONFIG
from .core.earai import EarAI, create_earai, InferenceResult
from .core.packets import VisionPacket, Entity, BBox, TextRegion, SceneChange
from .api.api import UniversalAPI, NativeAPI, StreamingAPI, APIResponse, create_streaming_api

__version__ = "0.1.0"
__all__ = [
    "EarAIConfig",
    "DEFAULT_CONFIG",
    "EarAI",
    "create_earai",
    "InferenceResult",
    "VisionPacket",
    "Entity",
    "BBox",
    "TextRegion",
    "SceneChange",
    "UniversalAPI",
    "NativeAPI",
    "StreamingAPI",
    "APIResponse",
    "create_streaming_api",
]