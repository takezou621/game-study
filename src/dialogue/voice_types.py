"""Data types for voice conversation functionality.

This module contains dataclasses and enums used across voice-related modules,
separated for better code organization and reusability.
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class SpeechState(Enum):
    """Current speech state."""

    IDLE = "idle"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class VoiceResponse:
    """Voice response from Realtime API."""

    text: str
    audio_data: bytes | None = None
    duration_ms: int | None = None
    timestamp: float = None
    priority: int = 2
    interrupted: bool = False

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class AudioChunk:
    """Audio chunk for playback queue."""

    data: bytes
    timestamp: float = field(default_factory=time.time)


# Short response templates for combat situations
COMBAT_TEMPLATES: dict[int, dict[str, str]] = {
    0: {  # P0 - Survival
        "low_hp": "Low HP! Cover!",
        "knocked": "Knocked! Ping!",
        "storm": "Storm! Move!",
    },
    1: {  # P1 - Tactical
        "rotate": "Rotate now!",
        "storm_shrinking": "Storm moving!",
    },
}
