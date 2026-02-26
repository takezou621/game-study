"""Audio and speech buffer management."""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioFrame:
    """Single audio frame."""

    data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    is_speech: bool = False

    @property
    def duration_ms(self) -> float:
        """Get frame duration in milliseconds."""
        return len(self.data) / 16.0  # Assuming 16kHz sample rate


@dataclass
class SpeechSegment:
    """Detected speech segment."""

    audio_data: np.ndarray
    start_time: float
    end_time: float
    confidence: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Get segment duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0

    @property
    def duration_seconds(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


class AudioBuffer:
    """Circular buffer for audio samples."""

    def __init__(self, max_size: int = 100):
        """
        Initialize audio buffer.

        Args:
            max_size: Maximum number of frames to buffer
        """
        self.max_size = max_size
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0

    def append(self, frame: np.ndarray) -> None:
        """
        Add frame to buffer.

        Args:
            frame: Audio frame data
        """
        if len(self._buffer) >= self.max_size:
            removed = self._buffer.pop(0)
            self._total_samples -= len(removed)

        self._buffer.append(frame)
        self._total_samples += len(frame)

    def get_all(self) -> np.ndarray:
        """
        Get all buffered audio.

        Returns:
            Concatenated audio data
        """
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self._buffer.clear()
        self._total_samples = 0

    @property
    def size(self) -> int:
        """Get number of buffered frames."""
        return len(self._buffer)

    @property
    def total_samples(self) -> int:
        """Get total number of samples."""
        return self._total_samples


class SpeechBuffer:
    """Buffer for accumulating speech segments."""

    def __init__(
        self,
        padding_ms: int = 300,
        min_speech_ms: int = 500,
        max_speech_ms: int = 10000,
        frame_duration_ms: float = 32.0,
    ):
        """
        Initialize speech buffer.

        Args:
            padding_ms: Padding before/after speech
            min_speech_ms: Minimum speech duration
            max_speech_ms: Maximum speech duration
            frame_duration_ms: Duration of each frame
        """
        self.padding_ms = padding_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.frame_duration_ms = frame_duration_ms

        # Calculate frame counts
        self.padding_frames = int(padding_ms / frame_duration_ms)
        self.min_speech_frames = int(min_speech_ms / frame_duration_ms)
        self.max_speech_frames = int(max_speech_ms / frame_duration_ms)

        # State
        self._buffer: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._start_time: float | None = None

    def add_frame(self, frame: np.ndarray, is_speech: bool) -> SpeechSegment | None:
        """
        Add frame and check for complete speech segment.

        Args:
            frame: Audio frame data
            is_speech: Whether frame contains speech

        Returns:
            SpeechSegment if segment is complete, None otherwise
        """
        if is_speech:
            return self._handle_speech_frame(frame)
        else:
            return self._handle_silence_frame(frame)

    def _handle_speech_frame(self, frame: np.ndarray) -> SpeechSegment | None:
        """Handle speech frame."""
        if not self._in_speech:
            # Start of speech
            self._in_speech = True
            self._start_time = time.time()
            self._buffer = []
            self._speech_frames = 0

        self._buffer.append(frame)
        self._speech_frames += 1
        self._silence_frames = 0

        # Check max duration
        if self._speech_frames >= self.max_speech_frames:
            return self.flush()

        return None

    def _handle_silence_frame(self, frame: np.ndarray) -> SpeechSegment | None:
        """Handle silence frame."""
        if not self._in_speech:
            return None

        # Continue to capture padding
        self._buffer.append(frame)
        self._silence_frames += 1

        # Check if padding complete
        if self._silence_frames >= self.padding_frames:
            # Check minimum duration
            if self._speech_frames >= self.min_speech_frames:
                return self.flush()

            # Reset if too short
            self._reset()

        return None

    def flush(self) -> SpeechSegment | None:
        """
        Flush buffer as speech segment.

        Returns:
            SpeechSegment if buffer has data, None otherwise
        """
        if not self._buffer:
            return None

        audio_data = np.concatenate(self._buffer)
        segment = SpeechSegment(
            audio_data=audio_data,
            start_time=self._start_time or time.time(),
            end_time=time.time(),
            confidence=0.8,  # Default confidence
        )

        self._reset()
        return segment

    def _reset(self) -> None:
        """Reset buffer state."""
        self._buffer = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._start_time = None

    @property
    def is_speech_active(self) -> bool:
        """Check if currently in speech."""
        return self._in_speech
