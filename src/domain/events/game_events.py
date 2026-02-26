"""Game-related domain events."""

from dataclasses import dataclass, field
from typing import Any

from domain.events.base import DomainEvent


@dataclass
class GameStateChanged(DomainEvent):
    """Event fired when game state changes."""

    event_type: str = "game_state_changed"
    previous_state: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    changed_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "previous_state": self.previous_state,
            "current_state": self.current_state,
            "changed_fields": self.changed_fields,
        }


@dataclass
class TriggerFired(DomainEvent):
    """Event fired when a trigger rule fires."""

    event_type: str = "trigger_fired"
    trigger_id: str = ""
    trigger_name: str = ""
    priority: int = 0
    template: str | None = None
    movement_state: str = "non_combat"
    game_state_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "trigger_id": self.trigger_id,
            "trigger_name": self.trigger_name,
            "priority": self.priority,
            "template": self.template,
            "movement_state": self.movement_state,
            "game_state_snapshot": self.game_state_snapshot,
        }


@dataclass
class PlayerStatusChanged(DomainEvent):
    """Event fired when player status changes."""

    event_type: str = "player_status_changed"
    hp_previous: int | None = None
    hp_current: int | None = None
    shield_previous: int | None = None
    shield_current: int | None = None
    is_knocked_previous: bool = False
    is_knocked_current: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "hp_previous": self.hp_previous,
            "hp_current": self.hp_current,
            "shield_previous": self.shield_previous,
            "shield_current": self.shield_current,
            "is_knocked_previous": self.is_knocked_previous,
            "is_knocked_current": self.is_knocked_current,
        }


@dataclass
class StormStatusChanged(DomainEvent):
    """Event fired when storm status changes."""

    event_type: str = "storm_status_changed"
    in_storm_previous: bool = False
    in_storm_current: bool = False
    is_shrinking_previous: bool = False
    is_shrinking_current: bool = False
    phase_previous: int | None = None
    phase_current: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "in_storm_previous": self.in_storm_previous,
            "in_storm_current": self.in_storm_current,
            "is_shrinking_previous": self.is_shrinking_previous,
            "is_shrinking_current": self.is_shrinking_current,
            "phase_previous": self.phase_previous,
            "phase_current": self.phase_current,
        }
