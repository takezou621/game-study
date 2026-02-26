"""Audio processing modules."""

from infrastructure.audio.processing.buffer import (
    AudioBuffer,
    AudioFrame,
    SpeechBuffer,
    SpeechSegment,
)
from infrastructure.audio.processing.noise_gate import NoiseGate, SimpleNoiseGate

__all__ = [
    "NoiseGate",
    "SimpleNoiseGate",
    "AudioBuffer",
    "SpeechBuffer",
    "AudioFrame",
    "SpeechSegment",
]
