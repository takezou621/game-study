"""Session entity."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.time import get_timestamp_ms


class SessionPhase(str, Enum):
    """Session phase enumeration."""

    LOBBY = "lobby"
    BUS = "bus"
    EARLY_GAME = "early_game"
    MID_GAME = "mid_game"
    LATE_GAME = "late_game"
    END_GAME = "end_game"


@dataclass
class Session:
    """Session entity.

    Represents a game session with phase tracking and inactivity.
    """

    phase: SessionPhase = SessionPhase.LOBBY
    inactivity_duration_ms: int = 0
    started_at_ms: int = field(default_factory=get_timestamp_ms)
    last_activity_ms: int = field(default_factory=get_timestamp_ms)

    @property
    def duration_ms(self) -> int:
        """Get session duration in milliseconds."""
        return get_timestamp_ms() - self.started_at_ms

    @property
    def is_inactive(self) -> bool:
        """Check if session is inactive (no activity for 30 seconds)."""
        return self.inactivity_duration_ms > 30000

    def record_activity(self) -> "Session":
        """Record activity and reset inactivity timer.

        Returns:
            New Session with updated activity timestamp
        """
        now = get_timestamp_ms()
        return Session(
            phase=self.phase,
            inactivity_duration_ms=0,
            started_at_ms=self.started_at_ms,
            last_activity_ms=now,
        )

    def update_inactivity(self) -> "Session":
        """Update inactivity duration.

        Returns:
            New Session with updated inactivity duration
        """
        now = get_timestamp_ms()
        return Session(
            phase=self.phase,
            inactivity_duration_ms=now - self.last_activity_ms,
            started_at_ms=self.started_at_ms,
            last_activity_ms=self.last_activity_ms,
        )

    def set_phase(self, new_phase: SessionPhase) -> "Session":
        """Set session phase.

        Args:
            new_phase: New session phase

        Returns:
            New Session with updated phase
        """
        return Session(
            phase=new_phase,
            inactivity_duration_ms=self.inactivity_duration_ms,
            started_at_ms=self.started_at_ms,
            last_activity_ms=self.last_activity_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "inactivity_duration_ms": self.inactivity_duration_ms,
            "started_at_ms": self.started_at_ms,
            "last_activity_ms": self.last_activity_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        phase_str = data.get("phase", "lobby")
        try:
            phase = SessionPhase(phase_str)
        except ValueError:
            phase = SessionPhase.LOBBY

        return cls(
            phase=phase,
            inactivity_duration_ms=data.get("inactivity_duration_ms", 0),
            started_at_ms=data.get("started_at_ms", get_timestamp_ms()),
            last_activity_ms=data.get("last_activity_ms", get_timestamp_ms()),
        )
