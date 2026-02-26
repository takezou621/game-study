"""Cooldown policy for trigger management."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CooldownPolicy:
    """Policy for managing trigger cooldowns."""

    default_cooldown_ms: int = 5000
    min_cooldown_ms: int = 1000
    max_cooldown_ms: int = 60000

    def calculate_cooldown(
        self,
        base_cooldown_ms: int,
        priority: int,
        _state: dict[str, Any],
    ) -> int:
        """Calculate effective cooldown based on priority and state.

        Args:
            base_cooldown_ms: Base cooldown from trigger rule
            priority: Trigger priority (0=highest)
            _state: Current game state (unused, kept for API consistency)

        Returns:
            Effective cooldown in milliseconds
        """
        # Higher priority triggers have shorter cooldowns
        priority_factor = 1.0 - (priority * 0.2)  # P0=1.0, P1=0.8, P2=0.6, P3=0.4
        effective_cooldown = int(base_cooldown_ms * priority_factor)

        # Clamp to valid range
        return max(self.min_cooldown_ms, min(self.max_cooldown_ms, effective_cooldown))

    def is_on_cooldown(
        self,
        last_triggered_ms: int,
        cooldown_ms: int,
        current_time_ms: int,
    ) -> bool:
        """Check if trigger is on cooldown.

        Args:
            last_triggered_ms: When trigger last fired
            cooldown_ms: Cooldown duration
            current_time_ms: Current time

        Returns:
            True if on cooldown
        """
        return (current_time_ms - last_triggered_ms) < cooldown_ms

    def get_remaining_cooldown(
        self,
        last_triggered_ms: int,
        cooldown_ms: int,
        current_time_ms: int,
    ) -> int:
        """Get remaining cooldown time.

        Args:
            last_triggered_ms: When trigger last fired
            cooldown_ms: Cooldown duration
            current_time_ms: Current time

        Returns:
            Remaining cooldown in milliseconds, 0 if not on cooldown
        """
        elapsed = current_time_ms - last_triggered_ms
        remaining = cooldown_ms - elapsed
        return max(0, remaining)
