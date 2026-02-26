"""Ports (interfaces) for external dependencies."""

from application.ports.audio_port import AudioConfig, AudioPort
from application.ports.capture_port import CaptureMetadata, CapturePort
from application.ports.llm_port import LLMPort, LLMResponse
from application.ports.tts_port import TTSConfig, TTSPort

__all__ = [
    "CapturePort",
    "CaptureMetadata",
    "LLMPort",
    "LLMResponse",
    "TTSPort",
    "TTSConfig",
    "AudioPort",
    "AudioConfig",
]
