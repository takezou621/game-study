"""Response DTOs for voice response pipeline."""

from dataclasses import dataclass, field
from typing import Any

from utils.time import get_timestamp_ms


@dataclass
class ResponseDTO:
    """Data transfer object for voice responses."""

    text: str
    priority: int = 2
    source_trigger: str | None = None
    movement_state: str = "non_combat"
    timestamp_ms: int = field(default_factory=get_timestamp_ms)
    interrupted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_short(self) -> bool:
        """Check if response is short (< 100 chars)."""
        return len(self.text) < 100

    @property
    def estimated_duration_ms(self) -> int:
        """Estimate response duration based on text length.

        Average speaking rate is ~150 words per minute.
        Average word length is ~5 characters.
        So ~12.5 characters per 100ms.
        """
        return int(len(self.text) / 12.5) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "priority": self.priority,
            "source_trigger": self.source_trigger,
            "movement_state": self.movement_state,
            "timestamp_ms": self.timestamp_ms,
            "interrupted": self.interrupted,
            "metadata": self.metadata,
        }


@dataclass
class AudioResponseDTO:
    """Data transfer object for audio responses."""

    text: str
    audio_data: bytes | None = None
    duration_ms: int | None = None
    priority: int = 2
    sample_rate: int = 24000
    source_trigger: str | None = None
    timestamp_ms: int = field(default_factory=get_timestamp_ms)
    interrupted: bool = False
    playback_started_ms: int | None = None
    playback_completed_ms: int | None = None

    @property
    def is_ready(self) -> bool:
        """Check if audio is ready for playback."""
        return self.audio_data is not None

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return (
            self.playback_started_ms is not None
            and self.playback_completed_ms is None
            and not self.interrupted
        )

    @property
    def playback_progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if not self.is_playing or self.duration_ms is None:
            return 0.0
        elapsed = get_timestamp_ms() - self.playback_started_ms
        return min(1.0, elapsed / self.duration_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without audio data)."""
        return {
            "text": self.text,
            "audio_data_size": len(self.audio_data) if self.audio_data else 0,
            "duration_ms": self.duration_ms,
            "priority": self.priority,
            "sample_rate": self.sample_rate,
            "source_trigger": self.source_trigger,
            "timestamp_ms": self.timestamp_ms,
            "interrupted": self.interrupted,
            "playback_started_ms": self.playback_started_ms,
            "playback_completed_ms": self.playback_completed_ms,
        }


@dataclass
class QueuedResponseDTO:
    """Response in the playback queue."""

    response: AudioResponseDTO
    queue_position: int = 0
    queued_at_ms: int = field(default_factory=get_timestamp_ms)

    @property
    def wait_time_ms(self) -> int:
        """Time spent waiting in queue."""
        return get_timestamp_ms() - self.queued_at_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response.to_dict(),
            "queue_position": self.queue_position,
            "queued_at_ms": self.queued_at_ms,
            "wait_time_ms": self.wait_time_ms,
        }
