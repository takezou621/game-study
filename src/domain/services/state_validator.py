"""State validator for game state business rules."""

from typing import Any

from domain.entities.game_state import GameState
from domain.value_objects.health import HP, Shield


class StateValidator:
    """Validator for game state business rules."""

    @staticmethod
    def validate_hp(value: int) -> bool:
        """Validate HP is within valid range."""
        try:
            HP(value=value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_shield(value: int) -> bool:
        """Validate Shield is within valid range."""
        try:
            Shield(value=value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_confidence(value: float) -> bool:
        """Validate confidence is within valid range."""
        return 0.0 <= value <= 1.0

    @staticmethod
    def validate_state_transition(
        _current: GameState,
        next_state: GameState,
    ) -> tuple[bool, str]:
        """Validate state transition is valid.

        Args:
            _current: Current game state (unused, kept for API consistency)
            next_state: Proposed next state

        Returns:
            Tuple of (is_valid, error_message)
        """
        # HP cannot go negative
        if next_state.player.status.hp.value < 0:
            return False, "HP cannot be negative"

        # HP cannot exceed max
        if next_state.player.status.hp.value > next_state.player.status.hp.max_value:
            return False, "HP cannot exceed max value"

        # Shield cannot go negative
        if next_state.player.status.shield.value < 0:
            return False, "Shield cannot be negative"

        # Shield cannot exceed max
        if next_state.player.status.shield.value > next_state.player.status.shield.max_value:
            return False, "Shield cannot exceed max value"

        # Ammo cannot go negative
        if next_state.player.weapon.ammo.value < 0:
            return False, "Ammo cannot be negative"

        # If knocked, cannot have HP > 0 (typical game rule)
        if next_state.player.status.is_knocked and next_state.player.status.hp.value > 0:
            # This is game-specific, might need adjustment
            pass  # Allow for now, depends on game mechanics

        return True, ""

    @staticmethod
    def validate_state_value(_value: Any, source: str, confidence: float) -> tuple[bool, str]:
        """Validate a state value with metadata.

        Args:
            _value: The value to validate (unused, kept for API consistency)
            source: Source of the value
            confidence: Confidence score

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not source:
            return False, "Source is required"

        if not StateValidator.validate_confidence(confidence):
            return False, f"Confidence must be between 0.0 and 1.0, got {confidence}"

        return True, ""

    @staticmethod
    def should_trigger_combat_response(state: GameState) -> bool:
        """Determine if combat response should be triggered.

        Args:
            state: Current game state

        Returns:
            True if combat response should be triggered
        """
        # Low HP
        if state.player.status.hp.is_low:
            return True

        # In storm
        if state.world.storm.in_storm:
            return True

        # Knocked
        return bool(state.player.status.is_knocked)

    @staticmethod
    def calculate_urgency(state: GameState) -> int:
        """Calculate urgency level (0-3) based on state.

        Args:
            state: Current game state

        Returns:
            Urgency level (0=low, 1=medium, 2=high, 3=critical)
        """
        urgency = 0

        # HP-based urgency
        if state.player.status.hp.is_critical:
            urgency = max(urgency, 3)
        elif state.player.status.hp.is_low:
            urgency = max(urgency, 2)

        # Storm urgency
        if state.world.storm.in_storm:
            if state.world.storm.damage and state.world.storm.damage > 5:
                urgency = max(urgency, 3)
            else:
                urgency = max(urgency, 2)

        # Knocked urgency
        if state.player.status.is_knocked:
            urgency = max(urgency, 3)

        return urgency
