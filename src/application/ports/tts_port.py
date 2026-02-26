"""TTS port interface for text-to-speech."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class VoiceStyle(str, Enum):
    """Voice style options."""

    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    URGENT = "urgent"
    CALM = "calm"


@dataclass(frozen=True)
class TTSConfig:
    """Configuration for TTS synthesis."""

    voice: str = "alloy"
    speed: float = 1.0
    style: VoiceStyle = VoiceStyle.NEUTRAL
    sample_rate: int = 24000


@dataclass(frozen=True)
class TTSResult:
    """Result of TTS synthesis."""

    audio_data: bytes
    duration_ms: int
    format: str = "pcm"
    sample_rate: int = 24000


@runtime_checkable
class TTSPort(Protocol):
    """Port (interface) for TTS functionality."""

    def synthesize(self, text: str, config: TTSConfig | None = None) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            config: Optional configuration override

        Returns:
            TTSResult with audio data

        Raises:
            TTSError: If synthesis fails
        """
        ...

    async def synthesize_stream(self, text: str, config: TTSConfig | None = None):
        """Stream synthesized speech.

        Args:
            text: Text to synthesize
            config: Optional configuration override

        Yields:
            Audio chunks

        Raises:
            TTSError: If synthesis fails
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        ...


class AsyncTTSPort(Protocol):
    """Asynchronous TTS port."""

    async def synthesize(self, text: str, config: TTSConfig | None = None) -> TTSResult: ...

    async def synthesize_stream(self, text: str, config: TTSConfig | None = None): ...

    @property
    def is_available(self) -> bool: ...
