"""Streaming correctness contracts: no synthetic detections or broken geometry."""
import numpy as np
import pytest
import torch
import cv2

from earai.core.config import EarAIConfig
from earai.core.earai import EarAI
from earai.core.packets import VisionPacket
from earai.heads.motion_compensation import MotionTransform
from earai.heads.residual_encoder import warp_frame
from earai.memory.scene_memory import SceneMemory


def test_affine_conversion_roundtrip_non_square_frames():
    pixel_affine = np.array([
        [0.98, -0.13, 12.0],
        [0.13, 0.98, -3.0],
    ], dtype=np.float32)
    size = np.array([320, 180], dtype=np.float32)
    motion = MotionTransform.from_pixel_affine(pixel_affine, 320, 180, 0.95)
    np.testing.assert_allclose(
        motion.to_pixel_affine(320, 180), pixel_affine, atol=1e-5
    )
    sample_norm = np.array([[0.2, 0.3], [0.7, 0.6]], dtype=np.float32)
    expected_pixel = (sample_norm * size) @ pixel_affine[:, :2].T + pixel_affine[:, 2]
    np.testing.assert_allclose(
        motion.warp_points(sample_norm) * size,
        expected_pixel, atol=1e-4
    )


def test_warp_frame_matches_open_cv_pixel_affine():
    height, width = 80, 160
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[15:45, 35:65] = (30, 160, 255)
    pixel_affine = np.array([
        [1.0, -0.05, 9.0],
        [0.05, 1.0, 4.0],
    ], dtype=np.float32)
    transform = MotionTransform.from_pixel_affine(
        pixel_affine, width, height
    )
    actual, valid = warp_frame(frame, transform)
    expected = cv2.warpAffine(
        frame, pixel_affine, (width, height), flags=cv2.INTER_LINEAR
    )
    assert actual.shape == frame.shape
    assert valid.shape == frame.shape[:2]
    assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 2


def test_untrained_token_decoder_does_not_fabricate_boxes():
    ear = EarAI.__new__(EarAI)
    result = ear._tokens_to_entities(
        torch.randn(1, 16, 32), np.full(16, 0.01)
    )
    assert result == []


def test_image_level_classification_does_not_become_tracked_object():
    config = EarAIConfig(backbone_pretrained=False)
    memory = SceneMemory(config)
    packet = VisionPacket(
        t=1.0,
        frame_id=1,
        scene_embedding=np.zeros(256, dtype=np.float32),
        entities=[],
        text_regions=[],
        changes=[],
        image_labels=[{
            "class": "tabby",
            "confidence": 0.65,
            "scope": "whole_image",
            "source_frame_id": 1,
        }],
    )
    enriched = memory.update(packet)
    output = enriched.to_universal_api()
    assert output["entities"] == []
    assert len(output["image_labels"]) == 1
    assert output["image_labels"][0]["scope"] == "whole_image"
    assert enriched.to_native_api()["image_labels"][0]["source_frame_id"] == 1


def test_ocr_accepts_fractional_tesseract_confidence(monkeypatch):
    import earai.core.earai as runtime

    if not runtime.TESSERACT_AVAILABLE:
        pytest.skip("pytesseract Python package not installed")
    def fake_image_to_data(image, output_type, lang):
        return {
            "text": ["TEST"],
            "conf": ["97.125"],
            "left": [10],
            "top": [10],
            "width": [50],
            "height": [20],
        }
    monkeypatch.setattr(runtime.pytesseract, "image_to_data", fake_image_to_data)
    ear = EarAI.__new__(EarAI)
    regions = ear._process_ocr_tesseract(
        np.zeros((100, 200, 3), dtype=np.uint8)
    )
    assert len(regions) == 1
    assert regions[0].confidence == pytest.approx(0.97125)
