#!/usr/bin/env python3
"""
EarAI Acceptance Tests
Reproducible benchmarks for Predictive Visual State streaming.

Run: PYTHONPATH=src python3 -m pytest src/earai/tests/test_acceptance.py -v
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from earai import EarAI, EarAIConfig, UniversalAPI


def make_test_frame(w=470, h=470):
    """Create a simple test frame with some content"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(frame, (350, 200), 80, (0, 255, 0), -1)
    cv2.putText(frame, "TEST", (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def count_backbone_calls(ear):
    """Wrap backbone forward to count calls"""
    original_forward = ear.backbone.forward
    count = [0]
    
    def counting_forward(*args, **kwargs):
        count[0] += 1
        return original_forward(*args, **kwargs)
    
    ear.backbone.forward = counting_forward
    return count


def test_static_scene():
    """Test 1: Static scene - 100 identical frames should be 1 KEYFRAME + 99 REUSE"""
    print("\n=== TEST: STATIC SCENE ===")
    
    config = EarAIConfig(backbone_pretrained=True)
    ear = EarAI(config, device='cpu')
    api = UniversalAPI(ear)
    
    frame = make_test_frame()
    
    backbone_calls = count_backbone_calls(ear)
    
    # Frame 1 - force full
    result = api.process_frame(frame, force_full=True)
    assert result.success, f"Frame 1 failed: {result.error}"
    assert result.data['meta']['decision'] == 'KEYFRAME', f"Frame 1: expected KEYFRAME, got {result.data['meta']['decision']}"
    
    # Frames 2-100 - should all be REUSE
    for i in range(2, 101):
        result = api.process_frame(frame)
        assert result.success, f"Frame {i} failed: {result.error}"
        assert result.data['meta']['decision'] == 'REUSE', f"Frame {i}: expected REUSE, got {result.data['meta']['decision']}"
    
    print(f"Backbone calls: {backbone_calls[0]} (expected 1)")
    assert backbone_calls[0] == 1, f"Expected 1 backbone call, got {backbone_calls[0]}"
    
    print("✓ STATIC test PASSED")
    return True


def test_moving_object():
    """Test 2: Small moving object - should use ROI_CORRECT for small changes"""
    print("\n=== TEST: MOVING OBJECT ===")
    
    config = EarAIConfig(backbone_pretrained=True)
    ear = EarAI(config, device='cpu')
    api = UniversalAPI(ear)
    
    frame = make_test_frame()
    
    # Frame 1
    result = api.process_frame(frame, force_full=True)
    assert result.success
    assert result.data['meta']['decision'] == 'KEYFRAME'
    
    # Frames 2-5: small object moving 10px per frame
    roi_correct_count = 0
    for i in range(2, 6):
        f = frame.copy()
        cv2.rectangle(f, (100 + (i-1)*10, 100), (120 + (i-1)*10, 120), (0, 255, 0), -1)
        result = api.process_frame(f)
        assert result.success, f"Frame {i} failed: {result.error}"
        # Should be ROI_CORRECT for small changes
        if result.data['meta']['decision'] == 'ROI_CORRECT':
            roi_correct_count += 1
        print(f"  Frame {i}: {result.data['meta']['decision']} (entities: {len(result.data['entities'])})")
    
    # At least one ROI_CORRECT expected for moving object
    assert roi_correct_count > 0, f"Expected at least one ROI_CORRECT, got {roi_correct_count}"
    
    print("✓ MOVING OBJECT test completed")
    return True


def test_camera_motion():
    """Test 3: Camera pan/zoom - should use REUSE with motion compensation"""
    print("\n=== TEST: CAMERA MOTION ===")
    
    config = EarAIConfig(backbone_pretrained=True)
    ear = EarAI(config, device='cpu')
    api = UniversalAPI(ear)
    
    frame = make_test_frame()
    
    # Frame 1
    result = api.process_frame(frame, force_full=True)
    assert result.success
    assert result.data['meta']['decision'] == 'KEYFRAME'
    
    # Frames 2-4: camera pan
    for i in range(2, 5):
        f = frame.copy()
        M = np.array([[1.0, 0.0, float(i*5)], [0.0, 1.0, float(i*3)]], dtype=np.float32)
        f = cv2.warpAffine(f, M, (frame.shape[1], frame.shape[0]))
        result = api.process_frame(f)
        assert result.success, f"Frame {i} failed: {result.error}"
        assert result.data['meta']['decision'] == 'REUSE', f"Frame {i}: expected REUSE, got {result.data['meta']['decision']}"
        print(f"  Frame {i}: {result.data['meta']['decision']} (entities: {len(result.data['entities'])})")
    
    print("✓ CAMERA MOTION test completed")
    return True


def test_scene_cut():
    """Test 4: Scene cut - should trigger KEYFRAME"""
    print("\n=== TEST: SCENE CUT ===")
    
    config = EarAIConfig(backbone_pretrained=True)
    ear = EarAI(config, device='cpu')
    api = UniversalAPI(ear)
    
    frame1 = make_test_frame()
    frame2 = np.zeros_like(frame1)
    cv2.rectangle(frame2, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.putText(frame2, "NEW SCENE", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Frame 1
    result1 = api.process_frame(frame1, force_full=True)
    assert result1.success
    assert result1.data['meta']['decision'] == 'KEYFRAME'
    
    # Frame 2 - completely different scene
    result2 = api.process_frame(frame2)
    assert result2.success
    # Should detect scene cut and do KEYFRAME
    assert result2.data['meta']['decision'] == 'KEYFRAME', f"Frame 2: expected KEYFRAME, got {result2.data['meta']['decision']}"
    print(f"  Frame 2 (scene cut): {result2.data['meta']['decision']}")
    
    print("✓ SCENE CUT test completed")
    return True


def test_text_persistence():
    """Test 5: Text persistence - OCR result should persist across REUSE frames"""
    print("\n=== TEST: TEXT PERSISTENCE ===")
    
    config = EarAIConfig(backbone_pretrained=True)
    ear = EarAI(config, device='cpu')
    api = UniversalAPI(ear)
    
    # Create frame with text
    frame = make_test_frame()
    cv2.putText(frame, "HELLO WORLD", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Frame 1 - should detect text
    result1 = api.process_frame(frame, force_full=True)
    assert result1.success
    text1 = [tr['value'] for tr in result1.data['text']]
    print(f"  Frame 1 text: {text1}")
    assert any("HELLO" in t or "WORLD" in t for t in text1), f"Expected HELLO/WORLD in text, got {text1}"
    
    # Frames 2-5 - same frame, should REUSE and keep text
    for i in range(2, 6):
        result = api.process_frame(frame)
        assert result.success, f"Frame {i} failed: {result.error}"
        assert result.data['meta']['decision'] == 'REUSE', f"Frame {i}: expected REUSE, got {result.data['meta']['decision']}"
        text = [tr['value'] for tr in result.data['text']]
        print(f"  Frame {i} text: {text}")
        # Text should persist
        assert any("HELLO" in t or "WORLD" in t for t in text), f"Frame {i}: text not persisted, got {text}"
    
    print("✓ TEXT PERSISTENCE test completed")
    return True


def run_all_tests():
    """Run all acceptance tests"""
    print("=" * 60)
    print("EarAI Acceptance Tests")
    print("=" * 60)
    
    tests = [
        ("STATIC", test_static_scene),
        ("MOVING_OBJECT", test_moving_object),
        ("CAMERA_MOTION", test_camera_motion),
        ("SCENE_CUT", test_scene_cut),
        ("TEXT_PERSISTENCE", test_text_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)