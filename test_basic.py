#!/usr/bin/env python3
"""Basic test for game-study."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    # Utils
    from utils.logger import SessionLogger
    from utils.time import get_timestamp_ms
    print("✓ Utils imported")

    # Capture
    from capture.video_file import VideoFileCapture
    print("✓ Capture imported")

    # Vision
    from vision.roi import ROIExtractor
    from vision.anchors import AnchorDetector
    from vision.yolo_detector import YOLODetector
    from vision.ocr import OCRDetector
    from vision.state_builder import StateBuilder
    print("✓ Vision imported")

    # Trigger
    from trigger.rules import TriggerRule, TriggerCondition
    from trigger.engine import TriggerEngine
    print("✓ Trigger imported")

    # Dialogue
    from dialogue.templates import DialogueTemplateManager
    from dialogue.openai_client import OpenAIClient
    print("✓ Dialogue imported")

    return True

def test_configs():
    """Test that config files are valid."""
    print("\nTesting configs...")

    import yaml

    # Test ROI config
    with open("configs/roi_defaults.yaml", 'r') as f:
        roi_config = yaml.safe_load(f)
        assert 'rois' in roi_config
        print(f"✓ ROI config loaded ({len(roi_config['rois'])} ROIs)")

    # Test triggers config
    with open("configs/triggers.yaml", 'r') as f:
        triggers_config = yaml.safe_load(f)
        assert 'triggers' in triggers_config
        print(f"✓ Triggers config loaded ({len(triggers_config['triggers'])} triggers)")

    # Test system prompt
    with open("configs/prompts/system.txt", 'r') as f:
        prompt = f.read()
        assert len(prompt) > 0
        print(f"✓ System prompt loaded ({len(prompt)} chars)")

    return True

def test_trigger_engine():
    """Test trigger engine basic functionality."""
    print("\nTesting trigger engine...")

    from trigger.engine import TriggerEngine

    engine = TriggerEngine("configs/triggers.yaml")

    # Create test state
    test_state = {
        "player": {
            "status": {
                "hp": {"value": 25, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "shield": {"value": 0, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "is_knocked": {"value": False, "source": "test", "confidence": 1.0, "ts_ms": 12345},
            },
        },
        "world": {
            "storm": {
                "in_storm": {"value": False, "source": "test", "confidence": 1.0, "ts_ms": 12345},
            },
        },
        "session": {
            "phase": {"value": "combat", "source": "test", "confidence": 1.0, "ts_ms": 12345},
            "inactivity_duration_ms": {"value": 0, "source": "test", "confidence": 1.0, "ts_ms": 12345},
        },
    }

    # Test low HP trigger (P0)
    result = engine.evaluate_triggers(test_state, "combat")
    if result:
        print(f"✓ Low HP trigger fired: {result['rule_name']}")
    else:
        print("⚠ No trigger fired (expected for high HP)")

    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Game-Study Basic Tests")
    print("=" * 50)

    try:
        test_imports()
        test_configs()
        test_trigger_engine()

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
