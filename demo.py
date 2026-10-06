#!/usr/bin/env python3
"""
Vera Eye Demo - Visual coprocessor 3-8 MB
Shows universal API (JSON for any LLM) and native API (embeddings)
"""
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, '/Users/earflow/vera-eye')

from vera_eye import (
    VeraEye, VeraEyeConfig, create_vera_eye,
    UniversalAPI, NativeAPI, StreamingAPI,
    VisionPacket
)


def demo_universal_api():
    """Demo: Universal JSON API - works with any LLM"""
    print("\n" + "="*60)
    print("DEMO: Universal API (JSON for any LLM)")
    print("="*60)

    config = VeraEyeConfig()
    vera = create_vera_eye(config, device="cpu")
    api = UniversalAPI(vera)

    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add some shapes
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), -1)
    cv2.circle(frame, (500, 200), 80, (0, 255, 0), -1)
    cv2.putText(frame, "TEST TEXT", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Process
    result = api.process_frame(frame, force_full=True)

    if result.success:
        print(f"\n✓ Frame {result.data['meta']['frame_id']} processed in {result.latency_ms:.1f}ms")
        print(f"  Mode: {result.data['meta']['inference_mode']}")

        # JSON output for LLM
        print("\n--- JSON for LLM ---")
        print(json.dumps(result.data, indent=2, default=str))

        # Natural language description
        print("\n--- Natural Language ---")
        # Need packet for description
        packet = vera.process_frame(frame, force_full=True).packet
        desc = api.get_scene_description(packet)
        print(desc)

        # Prompt context
        print("\n--- Prompt Context ---")
        print(api.to_prompt_context(packet))
    else:
        print(f"✗ Error: {result.error}")


def demo_native_api():
    """Demo: Native API - embeddings for trained models"""
    print("\n" + "="*60)
    print("DEMO: Native API (Embeddings for trained models)")
    print("="*60)

    config = VeraEyeConfig()
    vera = create_vera_eye(config, device="cpu")
    api = NativeAPI(vera)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (150, 150), (350, 350), (0, 0, 255), -1)

    result = api.process_frame(frame, force_full=True)

    if result.success:
        print(f"\n✓ Frame processed in {result.latency_ms:.1f}ms")

        # Entity embeddings
        entities = result.data.get("entities", [])
        print(f"\n  Entities: {len(entities)}")
        for e in entities[:3]:
            print(f"    #{e['id']}: {e['class_name']} conf={e['confidence']:.2f}")
            print(f"      visual_emb dim: {len(e['visual_embedding'])}")
            print(f"      semantic_emb dim: {len(e['semantic_embedding'])}")
            print(f"      motion: {e['motion_vector']}")

        # Scene embedding
        scene_emb = result.data.get("scene_embedding", [])
        print(f"\n  Scene embedding dim: {len(scene_emb)}")

        # World state
        world = result.data.get("world_state", {})
        print(f"\n  World state: {world.get('entity_count', 0)} confirmed entities")


def demo_streaming():
    """Demo: Streaming API with callback"""
    print("\n" + "="*60)
    print("DEMO: Streaming API (simulated)")
    print("="*60)

    config = VeraEyeConfig()
    frames_processed = []

    def on_frame(data):
        frames_processed.append(data)
        if len(frames_processed) <= 3:
            print(f"  Frame {data['meta']['frame_id']}: {data['scene_summary']}")

    streaming = StreamingAPI(config, device="cpu", api_mode="universal", on_frame=on_frame)

    # Simulate frames
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        if i == 2:
            cv2.rectangle(frame, (200, 200), (400, 400), (255, 255, 0), -1)
        streaming.vera_eye.process_frame(frame)

    print(f"\n  Processed {len(frames_processed)} frames")


def demo_persistent_memory():
    """Demo: Persistent scene memory across frames"""
    print("\n" + "="*60)
    print("DEMO: Persistent Scene Memory (World State)")
    print("="*60)

    config = VeraEyeConfig()
    vera = create_vera_eye(config, device="cpu")

    # Frame 1: Object appears
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (100, 100), (250, 250), (255, 0, 0), -1)
    cv2.putText(frame1, "RED BOX", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    r1 = vera.process_frame(frame1, force_full=True)
    print(f"\nFrame 1: {len(r1.packet.entities)} entities")
    for e in r1.packet.entities:
        print(f"  Created entity #{e.id}: {e.class_name} at {e.bbox.center()}")

    # Frame 2: Object moves
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (150, 150), (300, 300), (255, 0, 0), -1)

    r2 = vera.process_frame(frame2)
    print(f"\nFrame 2: {len(r2.packet.entities)} entities")
    for e in r2.packet.entities:
        print(f"  Entity #{e.id}: {e.class_name} at {e.bbox.center()} motion={e.motion_vector}")
    print(f"  Changes: {[c.type for c in r2.packet.changes]}")

    # Frame 3: Object continues moving
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame3, (200, 200), (350, 350), (255, 0, 0), -1)

    r3 = vera.process_frame(frame3)
    print(f"\nFrame 3: {len(r3.packet.entities)} entities")
    for e in r3.packet.entities:
        print(f"  Entity #{e.id}: {e.class_name} at {e.bbox.center()} motion={e.motion_vector}")
    print(f"  Changes: {[c.type for c in r3.packet.changes]}")

    # World state
    world = vera.scene_memory.get_world_state()
    print(f"\nWorld State: {world['entity_count']} confirmed entities")
    for e in world['entities']:
        print(f"  #{e['id']}: {e['class']} pos={e['position']} vel={e['velocity']} age={e['age']:.1f}s")


def demo_foveated_vision():
    """Demo: Foveated vision - peripheral + high-res crops"""
    print("\n" + "="*60)
    print("DEMO: Foveated Vision")
    print("="*60)

    config = VeraEyeConfig()
    vera = create_vera_eye(config, device="cpu")

    # Large frame with small text region
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Peripheral object
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), -1)
    # Small text in corner (needs fovea)
    cv2.putText(frame, "Tiny Text Here", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result = vera.process_frame(frame, force_full=True)

    print(f"\nPeripheral resolution: {config.peripheral_resolution}")
    print(f"Fovea resolution: {config.fovea_resolution}")
    print(f"Entities detected: {len(result.packet.entities)}")
    print(f"Fovea requests: {len(result.packet.fovea_requests)}")

    for req in result.packet.fovea_requests:
        print(f"  Fovea request: entity #{req['entity_id']} at {req['center']} zoom={req['zoom']} reason={req['reason']}")

    # Extract fovea crop manually
    if result.packet.fovea_requests:
        req = result.packet.fovea_requests[0]
        fovea = vera.foveated_vision.extract_fovea(frame, req)
        print(f"  Extracted fovea crop: {fovea.shape}")


def demo_size_report():
    """Report model size"""
    print("\n" + "="*60)
    print("MODEL SIZE REPORT")
    print("="*60)

    config = VeraEyeConfig()
    vera = create_vera_eye(config, device="cpu")

    backbone_params = sum(p.numel() for p in vera.backbone.parameters())
    heads_params = sum(
        sum(p.numel() for p in h.parameters())
        for h in vera.heads.values()
    )
    total_params = backbone_params + heads_params

    print(f"\nBackbone (MobileNetV3-Small): {backbone_params/1e6:.2f}M params")
    print(f"Heads: {heads_params/1e6:.2f}M params")
    print(f"Total: {total_params/1e6:.2f}M params")
    print(f"\nINT8 size: ~{total_params/1e6:.1f} MB")
    print(f"INT4 size: ~{total_params/2e6:.1f} MB")
    print(f"FP16 size: ~{total_params*2/1e6:.1f} MB")


if __name__ == "__main__":
    import json

    print("VERA EYE - Visual Coprocessor Demo")
    print("Target: 3-8 MB, persistent scene memory, foveated vision")

    demo_size_report()
    demo_universal_api()
    demo_native_api()
    demo_persistent_memory()
    demo_foveated_vision()
    demo_streaming()

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE")
    print("="*60)