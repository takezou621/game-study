"""Voice Activity Detection (VAD) module for speech detection.

This module provides VAD functionality using WebRTC VAD or silero-vad
to detect speech in audio streams.
"""

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VADError(Exception):
    """Base exception for VAD errors."""
    pass


class VADNotAvailableError(VADError):
    """Raised when VAD library is not available."""
    pass


class VADModel(Enum):
    """Available VAD models."""
    WEBRTC = "webrtc"
    SILERO = "silero"
    ENERGY = "energy"  # Simple energy-based fallback


@dataclass
class VADResult:
    """Result from VAD processing."""
    is_speech: bool
    confidence: float
    frame_duration_ms: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class VADConfig:
    """VAD configuration."""
    model: VADModel = VADModel.WEBRTC
    sample_rate: int = 16000
    frame_size_ms: int = 30  # Frame duration in ms
    threshold: float = 0.5  # Speech probability threshold
    use_silero: bool = False  # Prefer silero if available
    silero_model_path: Optional[str] = None  # Path to silero model


class VoiceActivityDetector:
    """
    Voice Activity Detector using multiple backends.

    Supports:
    - WebRTC VAD (webrtcvad package)
    - Silero VAD (silero-vad package)
    - Energy-based detection (fallback)

    Features:
    - Frame-level speech detection
    - Confidence scoring
    - Automatic backend selection
    - Thread-safe operation
    """

    # Frame sizes compatible with WebRTC VAD (10, 20, 30 ms)
    VALID_FRAME_SIZES = [10, 20, 30]

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size_ms: int = 30,
        model: Optional[VADModel] = None,
        threshold: float = 0.5
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000 for WebRTC)
            frame_size_ms: Frame duration in ms (10, 20, or 30 for WebRTC)
            model: VAD model to use (None for auto-detect)
            threshold: Speech probability threshold (0-1)
        """
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.threshold = threshold

        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_size_ms / 1000)

        # Backend
        self._model = model
        self._backend: Optional[str] = None
        self._vad: Optional[Any] = None
        self._silero_model: Optional[Any] = None
        self._lock = threading.Lock()

        # Energy-based VAD state
        self._energy_threshold = 0.01
        self._energy_adapt_rate = 0.1
        self._noise_estimate = 0.0

    def initialize(self) -> bool:
        """
        Initialize VAD backend.

        Returns:
            True if initialized successfully
        """
        with self._lock:
            # Try requested model first
            if self._model:
                if self._model == VADModel.SILERO:
                    if self._init_silero():
                        return True
                elif self._model == VADModel.WEBRTC:
                    if self._init_webrtc():
                        return True
                elif self._model == VADModel.ENERGY:
                    self._backend = "energy"
                    return True

            # Auto-detect best backend
            if self._init_silero():
                return True
            if self._init_webrtc():
                return True

            # Fallback to energy-based
            self._backend = "energy"
            logger.info("Using energy-based VAD")
            return True

    def process_frame(self, audio: np.ndarray) -> VADResult:
        """
        Process a single audio frame.

        Args:
            audio: Audio samples (float32, normalized -1 to 1, or int16)

        Returns:
            VADResult with speech detection
        """
        with self._lock:
            if self._backend == "silero":
                return self._process_silero(audio)
            elif self._backend == "webrtc":
                return self._process_webrtc(audio)
            else:
                return self._process_energy(audio)

    async def process_frame_async(self, audio: np.ndarray) -> VADResult:
        """Async version of process_frame."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_frame, audio)

    def process_stream(
        self,
        audio_stream: np.ndarray,
        frame_size_ms: Optional[int] = None
    ) -> List[VADResult]:
        """
        Process a stream of audio data.

        Args:
            audio_stream: Continuous audio samples
            frame_size_ms: Frame duration (None to use default)

        Returns:
            List of VAD results
        """
        frame_size = frame_size_ms or self.frame_size_ms
        frame_samples = int(self.sample_rate * frame_size / 1000)

        results = []
        for i in range(0, len(audio_stream), frame_samples):
            frame = audio_stream[i:i + frame_samples]

            # Pad if needed
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))

            results.append(self.process_frame(frame))

        return results

    def _init_webrtc(self) -> bool:
        """Initialize WebRTC VAD backend."""
        try:
            import webrtcvad

            # Validate sample rate
            if self.sample_rate not in [8000, 16000, 32000, 48000]:
                logger.warning(
                    f"WebRTC VAD requires sample rate of 8000, 16000, 32000, or 48000. "
                    f"Got {self.sample_rate}. Using energy-based VAD."
                )
                return False

            # Validate frame size
            if self.frame_size_ms not in self.VALID_FRAME_SIZES:
                logger.warning(
                    f"WebRTC VAD requires frame size of {self.VALID_FRAME_SIZES}ms. "
                    f"Got {self.frame_size_ms}. Using 30ms."
                )
                self.frame_size_ms = 30
                self.frame_size = int(self.sample_rate * 30 / 1000)

            # Create VAD instance (aggressiveness 0-3)
            self._vad = webrtcvad.Vad(2)  # Medium aggressiveness
            self._backend = "webrtc"
            logger.info("WebRTC VAD initialized")
            return True

        except ImportError:
            logger.debug("webrtcvad not available")
            return False

        except Exception as e:
            logger.warning(f"WebRTC VAD initialization failed: {e}")
            return False

    def _init_silero(self) -> bool:
        """Initialize Silero VAD backend."""
        try:
            from silero_vad import VadModel, utils

            # Download/load model
            model_path = self._load_silero_model()
            self._silero_model = VadModel.from_pretrained(model_path)
            self._backend = "silero"
            logger.info("Silero VAD initialized")
            return True

        except ImportError:
            logger.debug("silero-vad not available")
            return False

        except Exception as e:
            logger.warning(f"Silero VAD initialization failed: {e}")
            return False

    def _load_silero_model(self) -> str:
        """Get or download Silero model path."""
        # Check for cached model
        cache_dir = os.path.expanduser("~/.cache/silero-vad")
        os.makedirs(cache_dir, exist_ok=True)

        model_path = os.path.join(cache_dir, "silero_vad_v4.jit")

        if os.path.exists(model_path):
            return model_path

        # Download will be handled by silero-vad library
        return "silero_vad_v4"

    def _process_webrtc(self, audio: np.ndarray) -> VADResult:
        """Process frame with WebRTC VAD."""
        # Convert to int16 bytes
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        audio_bytes = audio_int16.tobytes()

        # WebRTC VAD returns boolean
        is_speech = self._vad.is_speech(audio_bytes, self.sample_rate)

        return VADResult(
            is_speech=is_speech,
            confidence=0.9 if is_speech else 0.1,  # WebRTC doesn't provide confidence
            frame_duration_ms=self.frame_size_ms
        )

    def _process_silero(self, audio: np.ndarray) -> VADResult:
        """Process frame with Silero VAD."""
        # Convert to float32 tensor
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Silero expects int16 -> float32 conversion
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        # Get speech probability
        speech_prob = self._silero_model(audio, self.sample_rate).item()

        is_speech = speech_prob > self.threshold

        return VADResult(
            is_speech=is_speech,
            confidence=speech_prob,
            frame_duration_ms=self.frame_size_ms
        )

    def _process_energy(self, audio: np.ndarray) -> VADResult:
        """Process frame with energy-based detection."""
        # Ensure float32 normalized
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))

        # Adaptive noise estimation
        self._noise_estimate = (
            self._energy_adapt_rate * rms +
            (1 - self._energy_adapt_rate) * self._noise_estimate
        )

        # Dynamic threshold (noise floor + margin)
        threshold = self._noise_estimate * 3.0 + self._energy_threshold

        is_speech = rms > threshold

        # Confidence based on ratio
        confidence = min(rms / threshold, 1.0) if threshold > 0 else 0.0

        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            frame_duration_ms=self.frame_size_ms
        )

    def reset(self) -> None:
        """Reset VAD state."""
        with self._lock:
            self._noise_estimate = 0.0


class StreamingVAD:
    """
    Streaming VAD for real-time speech detection.

    Maintains state across frames for more robust detection
    and provides speech segment detection.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        padding_ms: int = 300,
        min_speech_ms: int = 500,
        max_speech_ms: int = 10000
    ):
        """
        Initialize streaming VAD.

        Args:
            vad: Base VAD instance
            padding_ms: Padding to add before/after speech
            min_speech_ms: Minimum speech duration
            max_speech_ms: Maximum speech duration
        """
        self.vad = vad
        self.padding_ms = padding_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

        # State
        self._in_speech = False
        self._speech_start_time: Optional[float] = None
        self._silence_frames = 0
        self._speech_frames = 0

        # Buffers
        self._padding_frames = int(padding_ms / vad.frame_size_ms)
        self._min_speech_frames = int(min_speech_ms / vad.frame_size_ms)
        self._max_speech_frames = int(max_speech_ms / vad.frame_size_ms)

    def process_frame(self, audio: np.ndarray) -> VADResult:
        """
        Process frame with streaming state.

        Args:
            audio: Audio frame

        Returns:
            VADResult with streaming-aware speech detection
        """
        result = self.vad.process_frame(audio)

        if result.is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_time = time.time()
                self._speech_frames = 0
                self._silence_frames = 0

            self._speech_frames += 1

            # Check max duration
            if self._speech_frames > self._max_speech_frames:
                self._in_speech = False

        else:
            if self._in_speech:
                self._silence_frames += 1

                # End of speech after padding
                if self._silence_frames >= self._padding_frames:
                    self._in_speech = False

        return result

    def is_in_speech(self) -> bool:
        """Check if currently in speech segment."""
        return self._in_speech

    def get_speech_duration_ms(self) -> Optional[float]:
        """Get current speech segment duration in ms."""
        if self._speech_start_time is None:
            return None
        return (time.time() - self._speech_start_time) * 1000

    def reset(self) -> None:
        """Reset streaming state."""
        self._in_speech = False
        self._speech_start_time = None
        self._silence_frames = 0
        self._speech_frames = 0


def create_vad(
    sample_rate: int = 16000,
    model: Optional[VADModel] = None,
    threshold: float = 0.5
) -> VoiceActivityDetector:
    """
    Create a VAD instance with sensible defaults.

    Args:
        sample_rate: Audio sample rate
        model: VAD model preference (None for auto)
        threshold: Speech probability threshold

    Returns:
        Configured VoiceActivityDetector instance
    """
    vad = VoiceActivityDetector(
        sample_rate=sample_rate,
        model=model,
        threshold=threshold
    )

    vad.initialize()
    return vad
