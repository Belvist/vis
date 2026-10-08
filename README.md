# EarAI

EarAI is a research prototype for lightweight image and video perception. It combines a MobileNetV3-Small multi-scale backbone, adaptive visual tokens, change-aware updates, and scene memory. **It is not yet a verified general-purpose visual understanding model.**

## Code layout

- `src/earai/core` — streaming pipeline and data structures
- `src/earai/models` — multi-scale backbone
- `src/earai/heads` — visual token pooling, motion/change components, decoders
- `src/earai/memory` — scene memory
- `src/earai/training` — Gate 1 and Gate 2 training/data/losses
- `scripts/gate2_train.py` — screenshot-to-UI-structure experiment

## Install

Python 3.11 is recommended.

```bash
python -m pip install -e .
# For unit tests:
python -m pip install pytest scipy pillow
# For browser UI training (also downloads additional model weights):
python -m pip install -e '.[gate2]'
python -m playwright install chromium
```

`gate2` requires network access to capture public pages and load a CLIP teacher. The synthetic contract tests do not require a browser or pretrained model weights.

## Gate 2 — screenshot structure

```bash
python -m pytest src/earai/tests/test_gate2_contracts.py -q
python scripts/gate2_preflight.py --config configs/gate2.yaml --smoke
python scripts/gate2_train.py --config configs/gate2.yaml --device auto --regenerate-dataset
```

Data is split by website **domain** between train and validation. The browser records normalized bounding boxes and CSS metadata. Gate 2 trains on letterboxed input, so its box predictions and validation labels use **letterboxed image coordinates**, not the original screenshot coordinate system.

The gate2 configuration reserves 32 decoder queries for at most 24 targets. Query count, token positions and letterbox geometry form an explicit model contract. **Checkpoints trained under previous contracts (e.g. 20 queries or stretched-image/incorrect box labels) cannot be assumed compatible. Retrain Gate 2 and evaluate on unseen domains before claiming it works.**

## Streaming output contract

The runtime's MobileNet ImageNet head is **whole-image classification**; it does not predict per-object coordinates. Predictions are exported as `image_labels` with `scope: whole_image`, not as spatial `entities`. Untrained token updates no longer fabricate bounding boxes. Until a validated object-localization head is trained and integrated, `entities` can be empty even on a clearly populated image. Existing integrations that treated classifier labels as objects must read `image_labels` explicitly.

## Evidence and limitations

The contract tests check tensor shapes, gradient propagation, query capacity, consistent image/box geometry, CPU keyframes and camera-motion transforms. A successful CI run is **not** evidence of useful detection accuracy. They **do not** establish real-world recognition accuracy, UI semantic understanding, reading/OCR quality, video frame rate, model size or generalization. Production readiness requires reproducible runs on held-out scenes with published precision/recall/IoU, runtime latency, memory and checkpoint provenance.

Training data and large checkpoints are not committed to this repository. Reports, weights and benchmarks must be generated on the target hardware.
