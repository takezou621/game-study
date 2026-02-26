"""Audio-related domain events."""

from dataclasses import dataclass
from typing import Any

from domain.events.base import DomainEvent


@dataclass
class SpeechDetected(DomainEvent):
    """Event fired when speech is detected."""

    event_type: str = "speech_detected"
    duration_ms: int = 0
    confidence: float = 0.0
    audio_data_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "audio_data_size": self.audio_data_size,
        }


@dataclass
class AudioChunkEvent(DomainEvent):
    """Event for audio chunk processing."""

    event_type: str = "audio_chunk"
    chunk_size: int = 0
    sample_rate: int = 16000
    channels: int = 1
    is_speech: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "chunk_size": self.chunk_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "is_speech": self.is_speech,
        }


@dataclass
class SpeechInterrupted(DomainEvent):
    """Event fired when speech is interrupted."""

    event_type: str = "speech_interrupted"
    original_duration_ms: int = 0
    interrupted_at_ms: int = 0
    remaining_ms: int = 0
    priority: int = 0  # Priority of interrupting trigger

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            **super().to_dict(),
            "original_duration_ms": self.original_duration_ms,
            "interrupted_at_ms": self.interrupted_at_ms,
            "remaining_ms": self.remaining_ms,
            "priority": self.priority,
        }
