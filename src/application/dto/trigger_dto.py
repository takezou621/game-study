"""Trigger DTOs for trigger evaluation pipeline."""

from dataclasses import dataclass, field
from typing import Any

from utils.time import get_timestamp_ms


@dataclass
class TriggerEvaluationDTO:
    """Input for trigger evaluation."""

    state: dict[str, Any]
    movement_state: str = "non_combat"
    evaluation_time_ms: int = field(default_factory=get_timestamp_ms)
    active_speech_priority: int | None = None  # Current playing trigger priority

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "movement_state": self.movement_state,
            "evaluation_time_ms": self.evaluation_time_ms,
            "active_speech_priority": self.active_speech_priority,
        }


@dataclass
class TriggerResultDTO:
    """Result of trigger evaluation."""

    trigger_id: str
    trigger_name: str
    priority: int
    template: str | None
    movement_state: str
    should_interrupt: bool = False
    confidence: float = 1.0
    evaluation_time_ms: int = field(default_factory=get_timestamp_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trigger_id": self.trigger_id,
            "trigger_name": self.trigger_name,
            "priority": self.priority,
            "template": self.template,
            "movement_state": self.movement_state,
            "should_interrupt": self.should_interrupt,
            "confidence": self.confidence,
            "evaluation_time_ms": self.evaluation_time_ms,
        }


@dataclass
class TriggerEvaluationResultDTO:
    """Complete result of trigger evaluation."""

    firing_triggers: list[TriggerResultDTO] = field(default_factory=list)
    selected_trigger: TriggerResultDTO | None = None
    suppressed_count: int = 0
    evaluation_time_ms: int = field(default_factory=get_timestamp_ms)

    @property
    def has_firing_triggers(self) -> bool:
        """Check if any triggers fired."""
        return len(self.firing_triggers) > 0

    @property
    def highest_priority(self) -> int | None:
        """Get highest priority among firing triggers."""
        if not self.firing_triggers:
            return None
        return min(t.priority for t in self.firing_triggers)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "firing_triggers": [t.to_dict() for t in self.firing_triggers],
            "selected_trigger": self.selected_trigger.to_dict() if self.selected_trigger else None,
            "suppressed_count": self.suppressed_count,
            "evaluation_time_ms": self.evaluation_time_ms,
        }
