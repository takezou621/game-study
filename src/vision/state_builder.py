"""Game State builder from vision detections."""

from typing import Dict, Any, Optional
from ..utils.time import get_timestamp_ms


class StateBuilder:
    """Build Game State JSON from vision detections."""

    def __init__(self):
        """Initialize state builder."""
        self.current_state = self._create_empty_state()

    def _create_empty_state(self) -> Dict[str, Any]:
        """
        Create empty game state template.

        Returns:
            Empty state dictionary
        """
        return {
            "player": {
                "status": {
                    "hp": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                    "shield": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                    "is_knocked": {"value": False, "source": None, "confidence": 0.0, "ts_ms": None},
                },
                "weapon": {
                    "name": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                    "ammo": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                },
                "inventory": {
                    "materials": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                },
            },
            "world": {
                "storm": {
                    "phase": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                    "damage": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                    "in_storm": {"value": False, "source": None, "confidence": 0.0, "ts_ms": None},
                    "is_shrinking": {"value": False, "source": None, "confidence": 0.0, "ts_ms": None},
                    "next_circle_distance": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                },
            },
            "session": {
                "phase": {"value": None, "source": None, "confidence": 0.0, "ts_ms": None},
                "inactivity_duration_ms": {"value": 0, "source": None, "confidence": 1.0, "ts_ms": None},
            },
        }

    def update_field(
        self,
        path: str,
        value: Any,
        source: str,
        confidence: float
    ) -> None:
        """
        Update a specific field in the state.

        Args:
            path: Dot-separated path (e.g., "player.status.hp")
            value: New value
            source: Source of the value (roi, yolo, ocr, etc.)
            confidence: Confidence score (0.0-1.0)
        """
        keys = path.split('.')
        current = self.current_state

        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]

        last_key = keys[-1]
        if last_key in current:
            # Update the field
            if isinstance(current[last_key], dict):
                current[last_key]["value"] = value
                current[last_key]["source"] = source
                current[last_key]["confidence"] = confidence
                current[last_key]["ts_ms"] = get_timestamp_ms()

    def update_hp(self, value: int, source: str, confidence: float) -> None:
        """Update HP value."""
        self.update_field("player.status.hp", value, source, confidence)

    def update_shield(self, value: int, source: str, confidence: float) -> None:
        """Update Shield value."""
        self.update_field("player.status.shield", value, source, confidence)

    def update_knocked(self, value: bool, source: str, confidence: float) -> None:
        """Update knocked status."""
        self.update_field("player.status.is_knocked", value, source, confidence)

    def update_weapon_name(self, value: str, source: str, confidence: float) -> None:
        """Update weapon name."""
        self.update_field("player.weapon.name", value, source, confidence)

    def update_ammo(self, value: int, source: str, confidence: float) -> None:
        """Update ammo count."""
        self.update_field("player.weapon.ammo", value, source, confidence)

    def update_materials(self, value: int, source: str, confidence: float) -> None:
        """Update materials count."""
        self.update_field("player.inventory.materials", value, source, confidence)

    def update_storm_phase(self, value: int, source: str, confidence: float) -> None:
        """Update storm phase."""
        self.update_field("world.storm.phase", value, source, confidence)

    def update_storm_damage(self, value: float, source: str, confidence: float) -> None:
        """Update storm damage."""
        self.update_field("world.storm.damage", value, source, confidence)

    def update_in_storm(self, value: bool, source: str, confidence: float) -> None:
        """Update in-storm status."""
        self.update_field("world.storm.in_storm", value, source, confidence)

    def update_storm_shrinking(self, value: bool, source: str, confidence: float) -> None:
        """Update storm shrinking status."""
        self.update_field("world.storm.is_shrinking", value, source, confidence)

    def update_session_phase(self, value: str, source: str, confidence: float) -> None:
        """Update session phase."""
        self.update_field("session.phase", value, source, confidence)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current state.

        Returns:
            Current game state
        """
        return self.current_state

    def reset(self) -> None:
        """Reset state to empty template."""
        self.current_state = self._create_empty_state()

    def get_movement_state(self) -> str:
        """
        Determine movement state based on current state.

        Returns:
            Movement state: "combat" or "non_combat"
        """
        # Heuristic: if HP < 50 or in storm, consider combat
        hp = self.current_state["player"]["status"]["hp"]["value"]
        in_storm = self.current_state["world"]["storm"]["in_storm"]["value"]

        if (hp is not None and hp < 50) or in_storm:
            return "combat"
        return "non_combat"
