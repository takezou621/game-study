"""Audio port interface for audio capture and playback."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class AudioDeviceType(str, Enum):
    """Audio device type."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class AudioDevice:
    """Information about an audio device."""

    id: int | str
    name: str
    type: AudioDeviceType
    sample_rate: int
    channels: int
    is_default: bool = False


@dataclass
class AudioConfig:
    """Configuration for audio capture/playback."""

    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 1024
    dtype: str = "float32"


@dataclass(frozen=True)
class AudioFrame:
    """Single audio frame."""

    data: bytes
    timestamp_ms: int
    is_speech: bool = False
    energy: float = 0.0


@runtime_checkable
class AudioPort(Protocol):
    """Port (interface) for audio functionality."""

    def list_devices(self) -> list[AudioDevice]:
        """List available audio devices.

        Returns:
            List of AudioDevice objects
        """
        ...

    def open_input(
        self,
        device_id: int | str | None = None,
        config: AudioConfig | None = None,
    ) -> None:
        """Open audio input stream.

        Args:
            device_id: Device ID or None for default
            config: Audio configuration

        Raises:
            AudioError: If stream cannot be opened
        """
        ...

    def read_frame(self) -> AudioFrame | None:
        """Read a single audio frame.

        Returns:
            AudioFrame or None if no data available
        """
        ...

    def close_input(self) -> None:
        """Close input stream."""
        ...

    @property
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        ...


class AudioPlaybackPort(Protocol):
    """Port for audio playback."""

    def open_output(
        self,
        device_id: int | str | None = None,
        config: AudioConfig | None = None,
    ) -> None:
        """Open audio output stream.

        Args:
            device_id: Device ID or None for default
            config: Audio configuration
        """
        ...

    def write_frame(self, frame: AudioFrame) -> None:
        """Write a frame to output.

        Args:
            frame: Audio frame to play
        """
        ...

    def close_output(self) -> None:
        """Close output stream."""
        ...

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        ...


class AsyncAudioPort(Protocol):
    """Asynchronous audio port."""

    async def open_input(
        self,
        device_id: int | str | None = None,
        config: AudioConfig | None = None,
    ) -> None: ...

    async def read_frame(self) -> AudioFrame | None: ...

    async def close_input(self) -> None: ...

    @property
    def is_capturing(self) -> bool: ...
