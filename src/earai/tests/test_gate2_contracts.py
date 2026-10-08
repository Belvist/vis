"""Synthetic Gate 2 contract tests; no browser, network or pretrained weights."""
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from earai.heads.adaptive_tokens import MultiScaleTokenLearner
from earai.training.browser_dataset import _manifest_signature, _manifest_valid
from earai.training.gate2_cache import (
    CachedGate2Dataset,
    _box_to_letterbox,
    _resize_with_padding,
)
from earai.training.gate2_decoder import Gate2Decoder
from earai.training.gate2_student import create_gate2_student

ROOT = Path(__file__).resolve().parents[3]


def test_letterbox_landscape_portrait_and_edge_boxes():
    landscape = Image.new("RGB", (400, 200))
    _, geometry = _resize_with_padding(landscape, (224, 224))
    assert geometry == (0, 56, 224, 112)
    assert _box_to_letterbox([0, 0, 1, 1], geometry, (224, 224)) == [
        0, 0.25, 1, 0.75
    ]

    portrait = Image.new("RGB", (200, 400))
    _, geometry = _resize_with_padding(portrait, (224, 224))
    assert geometry == (56, 0, 112, 224)
    assert _box_to_letterbox([0, 0, 1, 1], geometry, (224, 224)) == [
        0.25, 0, 0.75, 1
    ]


def test_cached_targets_and_image_use_same_letterbox_coordinates(tmp_path):
    image_path = tmp_path / "ui.png"
    Image.new("RGB", (400, 200)).save(image_path)
    cache = {
        "targets": [{
            "image_path": str(image_path),
            "image_id": "example",
            "split": "train",
            "ui_elements": [{
                "element_id": "one",
                "parent_id": None,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "class_id": 5,
                "style": {},
            }],
        }],
        "clip_embeddings": torch.zeros(1, 512),
    }
    dataset = CachedGate2Dataset(
        cache, {"image_size": [224, 224], "max_objects": 2}, "train"
    )
    item = dataset[0]
    assert item["student"].shape == (3, 224, 224)
    assert torch.allclose(
        item["boxes"][0], torch.tensor([0.0, 0.25, 1.0, 0.75])
    )
    # The original browser target must remain in source-image coordinates.
    assert cache["targets"][0]["ui_elements"][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_manifest_settings_and_missing_images_invalidate_cache(tmp_path):
    a = _manifest_signature(["https://example.org"], [(224, 224)], [0.0], 24)
    b = _manifest_signature(["https://example.org"], [(375, 667)], [0.0], 24)
    c = _manifest_signature(["https://example.org"], [(224, 224)], [0.0], 32)
    assert len({a, b, c}) == 3
    assert not _manifest_valid([{
        "split": "train",
        "domain": "example.org",
        "image_path": str(tmp_path / "deleted.png"),
        "ui_elements": [{"element_id": "x", "parent_id": None}],
    }], "train", {"example.org"})


def test_multiscale_token_centroids_and_degenerate_feature_map():
    torch.manual_seed(7)
    pooler = MultiScaleTokenLearner(
        channels_list=[8, 8, 8, 8],
        num_tokens_per_scale=[2, 2, 2, 2],
        bottleneck_dim=8,
    )
    out = pooler([
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 8, 4, 4),
        torch.randn(2, 8, 2, 2),
        torch.randn(2, 8, 1, 1),
    ], return_attention=True)
    assert out["tokens"].shape == (2, 8, 8)
    assert out["centroids"].shape == (2, 8, 2)
    assert torch.isfinite(out["centroids"]).all()
    assert ((out["centroids"] >= 0) & (out["centroids"] <= 1)).all()


def test_decoder_responds_to_centroids_and_backpropagates_to_position():
    torch.manual_seed(0)
    decoder = Gate2Decoder(token_dim=16, hidden_dim=16, num_queries=8)
    tokens = torch.randn(1, 8, 16)
    coords = torch.zeros(1, 8, 2, requires_grad=True)
    out_a = decoder(tokens, centroids=coords)
    out_b = decoder(tokens, centroids=torch.ones_like(coords))
    assert not torch.allclose(out_a["class_logits"], out_b["class_logits"])
    out_a["class_logits"].square().mean().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert decoder.object_decoder.centroid_proj.weight.grad is not None


def test_gate2_query_budget_rejects_impossible_recall():
    with pytest.raises(ValueError, match="must exceed max_objects"):
        create_gate2_student({
            "backbone_pretrained": False,
            "feature_dim": 32,
            "num_scene_tokens": 20,
            "max_objects": 24,
        })


def test_gate2_model_spatial_contract_and_gradient():
    model = create_gate2_student({
        "backbone_pretrained": False,
        "feature_dim": 32,
        "hidden_dim": 32,
        "num_scene_tokens": 8,
        "max_objects": 6,
    })
    model.eval()
    out = model(torch.randn(1, 3, 128, 128))
    assert out["tokens"].shape == (1, 8, 32)
    assert out["centroids"].shape == (1, 8, 2)
    assert out["bboxes_xyxy"].shape == (1, 8, 4)
    assert torch.isfinite(out["bboxes_xyxy"]).all()
    out["class_logits"].square().mean().backward()
    grad = model.decoder.object_decoder.centroid_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_production_gate2_config_reserves_background_queries():
    with (ROOT / "configs/gate2.yaml").open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["num_scene_tokens"] > cfg["max_objects"]
    assert cfg["num_scene_tokens"] - cfg["max_objects"] >= 1
