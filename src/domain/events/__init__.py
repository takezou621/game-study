"""Domain events for inter-module communication."""

from domain.events.audio_events import (
    AudioChunkEvent,
    SpeechDetected,
    SpeechInterrupted,
)
from domain.events.base import DomainEvent
from domain.events.game_events import (
    GameStateChanged,
    PlayerStatusChanged,
    StormStatusChanged,
    TriggerFired,
)

__all__ = [
    "DomainEvent",
    "GameStateChanged",
    "TriggerFired",
    "PlayerStatusChanged",
    "StormStatusChanged",
    "SpeechDetected",
    "AudioChunkEvent",
    "SpeechInterrupted",
]
