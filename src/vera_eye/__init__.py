"""Vera Eye - Ultra-lightweight visual coprocessor (3-8 MB)"""

from .core.config import VeraEyeConfig, DEFAULT_CONFIG
from .core.vera_eye import VeraEye, create_vera_eye, InferenceResult
from .core.packets import VisionPacket, Entity, BBox, TextRegion, SceneChange
from .api.api import UniversalAPI, NativeAPI, StreamingAPI, APIResponse, create_streaming_api

__version__ = "0.1.0"
__all__ = [
    "VeraEyeConfig",
    "DEFAULT_CONFIG",
    "VeraEye",
    "create_vera_eye",
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