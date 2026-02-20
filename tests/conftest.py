"""Common fixtures for tests."""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Game State Fixtures
# ============================================================================

@pytest.fixture
def base_state() -> Dict[str, Any]:
    """Create a basic game state for testing."""
    return {
        "player": {
            "status": {
                "hp": {"value": 100, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "shield": {"value": 100, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "is_knocked": {"value": False, "source": "test", "confidence": 1.0, "ts_ms": 12345},
            },
            "weapon": {
                "name": {"value": "Assault Rifle", "source": "test", "confidence": 0.9, "ts_ms": 12345},
                "ammo": {"value": 30, "source": "test", "confidence": 0.9, "ts_ms": 12345},
            },
            "inventory": {
                "materials": {"value": 500, "source": "test", "confidence": 0.8, "ts_ms": 12345},
            },
        },
        "world": {
            "storm": {
                "phase": {"value": 1, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "damage": {"value": 1.0, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "in_storm": {"value": False, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "is_shrinking": {"value": False, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                "next_circle_distance": {"value": 100.0, "source": "test", "confidence": 0.8, "ts_ms": 12345},
            },
        },
        "session": {
            "phase": {"value": "combat", "source": "test", "confidence": 0.9, "ts_ms": 12345},
            "inactivity_duration_ms": {"value": 0, "source": "test", "confidence": 1.0, "ts_ms": 12345},
        },
    }


@pytest.fixture
def low_hp_state(base_state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a game state with low HP."""
    state = base_state.copy()
    state["player"]["status"]["hp"]["value"] = 25
    return state


@pytest.fixture
def in_storm_state(base_state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a game state where player is in storm."""
    state = base_state.copy()
    state["world"]["storm"]["in_storm"]["value"] = True
    return state


@pytest.fixture
def knocked_state(base_state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a game state where player is knocked."""
    state = base_state.copy()
    state["player"]["status"]["is_knocked"]["value"] = True
    state["player"]["status"]["hp"]["value"] = 0
    return state


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path):
    """Create a temporary config file."""
    config_content = """
triggers:
  - id: test_trigger
    name: Test Trigger
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Low HP! Heal!"
      non_combat: "Your HP is low."
    cooldown_ms: 5000
"""
    config_path = temp_dir / "test_triggers.yaml"
    config_path.write_text(config_content)
    return config_path


# ============================================================================
# Module Fixtures
# ============================================================================

@pytest.fixture
def state_builder():
    """Create a StateBuilder instance."""
    from vision.state_builder import StateBuilder
    return StateBuilder()


@pytest.fixture
def trigger_condition():
    """Create a sample trigger condition."""
    from trigger.rules import TriggerCondition
    return TriggerCondition(
        field="player.status.hp",
        operator="lt",
        value=30
    )


@pytest.fixture
def trigger_rule():
    """Create a sample trigger rule."""
    from trigger.rules import TriggerRule, TriggerCondition

    conditions = [
        TriggerCondition(field="player.status.hp", operator="lt", value=30)
    ]
    templates = {
        "combat": "Low HP! Heal!",
        "non_combat": "Your HP is low."
    }

    return TriggerRule(
        rule_id="test_low_hp",
        name="Low HP Alert",
        priority=0,
        enabled=True,
        conditions=conditions,
        templates=templates,
        cooldown_ms=5000
    )


@pytest.fixture
def trigger_engine(temp_config_file: Path):
    """Create a TriggerEngine instance with test config."""
    from trigger.engine import TriggerEngine
    return TriggerEngine(str(temp_config_file))


@pytest.fixture
def session_logger(temp_dir: Path):
    """Create a SessionLogger instance."""
    from utils.logger import SessionLogger
    return SessionLogger(str(temp_dir))


@pytest.fixture
def webrtc_server():
    """Create a WebRTCSignalingServer instance."""
    from utils.webrtc import WebRTCSignalingServer
    return WebRTCSignalingServer(host="127.0.0.1", port=8080)
