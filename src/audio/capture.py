"""Audio capture module for microphone input and voice activity detection.

This module provides functionality for capturing audio from the microphone,
detecting voice activity, and managing audio streams for speech recognition.
"""

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from audio.vad import VoiceActivityDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class CaptureState(Enum):
    """Audio capture state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 16000  # 16kHz for speech recognition
    channels: int = 1  # Mono
    chunk_size: int = 512  # Frames per chunk
    format: str = "int16"  # Audio format
    device_index: int | None = None  # None for default device
    noise_gate_threshold: float = 0.01  # Noise gate (0-1)
    noise_gate_attack_ms: float = 5.0  # Attack time in ms
    noise_gate_release_ms: float = 50.0  # Release time in ms
    vad_enabled: bool = True  # Enable VAD
    vad_padding_ms: int = 300  # Padding before/after speech
    vad_min_speech_ms: int = 500  # Minimum speech duration
    vad_max_speech_ms: int = 10000  # Maximum speech duration


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


class AudioCaptureError(Exception):
    """Base exception for audio capture errors."""
    pass


class DeviceNotFoundError(AudioCaptureError):
    """Raised when audio device is not found."""
    pass


class AudioCapture:
    """
    Audio capture manager for microphone input.

    Handles audio device management, streaming capture, noise gating,
    and coordinates with VAD for speech detection.

    Features:
    - Cross-platform audio capture (sounddevice/pyaudio)
    - Noise gate with configurable attack/release
    - VAD integration for speech detection
    - Async/callback-based audio delivery
    - Thread-safe operation
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        on_frame_callback: Callable[[AudioFrame], None] | None = None,
        on_speech_callback: Callable[[SpeechSegment], None] | None = None
    ):
        """
        Initialize audio capture.

        Args:
            config: Audio configuration
            on_frame_callback: Callback for each audio frame
            on_speech_callback: Callback when speech segment detected
        """
        self.config = config or AudioConfig()
        self.on_frame_callback = on_frame_callback
        self.on_speech_callback = on_speech_callback

        # State
        self.state = CaptureState.STOPPED
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Audio libraries
        self._sd: Any | None = None
        self._pyaudio: Any | None = None
        self._stream: Any | None = None

        # VAD
        self._vad: VoiceActivityDetector | None = None
        self._vad_enabled = self.config.vad_enabled

        # Speech detection buffers
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_time: float | None = None
        self._silence_frames = 0
        self._speech_frames = 0
        self._in_speech = False

        # Noise gate state
        self._gate_open = False
        self._gate_envelope = 0.0

        # Queue for async delivery
        self._frame_queue: queue.Queue = queue.Queue(maxsize=100)
        self._speech_queue: queue.Queue = queue.Queue(maxsize=10)

    def initialize(self) -> bool:
        """
        Initialize audio capture system.

        Returns:
            True if initialized successfully
        """
        try:
            # Try sounddevice first, then pyaudio
            self._sd = self._try_import_sounddevice()
            if not self._sd:
                self._pyaudio = self._try_import_pyaudio()

            if not self._sd and not self._pyaudio:
                raise AudioCaptureError(
                    "No audio library available. Install sounddevice or pyaudio."
                )

            # Initialize VAD if enabled
            if self._vad_enabled:
                self._vad = VoiceActivityDetector(
                    sample_rate=self.config.sample_rate
                )
                if not self._vad.initialize():
                    logger.warning("VAD initialization failed, continuing without VAD")
                    self._vad_enabled = False
                else:
                    logger.info("VAD initialized successfully")

            self.state = CaptureState.STOPPED
            logger.info("Audio capture initialized")
            return True

        except Exception as e:
            logger.error(f"Audio capture initialization failed: {e}")
            self.state = CaptureState.ERROR
            return False

    def start(self) -> bool:
        """
        Start audio capture.

        Returns:
            True if started successfully
        """
        if self.state == CaptureState.RUNNING:
            return True

        self.state = CaptureState.STARTING
        self._stop_event.clear()

        try:
            if self._sd:
                return self._start_sounddevice()
            elif self._pyaudio:
                return self._start_pyaudio()
            else:
                raise AudioCaptureError("No audio library available")

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.state = CaptureState.ERROR
            return False

    def stop(self) -> None:
        """Stop audio capture."""
        if self.state == CaptureState.STOPPED:
            return

        self.state = CaptureState.STOPPING
        self._stop_event.set()

        # Flush any remaining speech
        if self._in_speech and self._speech_buffer:
            self._flush_speech_buffer()

        # Close stream
        if self._stream:
            try:
                if self._sd:
                    self._stream.stop()
                    self._stream.close()
                elif self._pyaudio:
                    self._stream.stop_stream()
                    self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")

        # Wait for thread
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._stream = None
        self.state = CaptureState.STOPPED
        logger.info("Audio capture stopped")

    def read_frame(self, timeout: float = 0.1) -> AudioFrame | None:
        """
        Read a single audio frame (blocking).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            AudioFrame or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_speech_segment(self, timeout: float = 1.0) -> SpeechSegment | None:
        """
        Read a detected speech segment (blocking).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            SpeechSegment or None if timeout
        """
        try:
            return self._speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def read_frame_async(self) -> AudioFrame | None:
        """Async version of read_frame."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_frame)

    async def read_speech_segment_async(self) -> SpeechSegment | None:
        """Async version of read_speech_segment."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_speech_segment)

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """
        List available audio input devices.

        Returns:
            List of device information dictionaries
        """
        devices = []

        # Try sounddevice
        try:
            import sounddevice as sd
            for i, info in enumerate(sd.query_devices()):
                if info.get('max_input_channels', 0) > 0:
                    devices.append({
                        'index': i,
                        'name': info.get('name', f'Device {i}'),
                        'channels': info.get('max_input_channels', 0),
                        'sample_rate': info.get('default_samplerate', 0)
                    })
        except ImportError:
            pass

        # Try pyaudio
        if not devices:
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    if info.get('maxInputChannels', 0) > 0:
                        devices.append({
                            'index': i,
                            'name': info.get('name', f'Device {i}'),
                            'channels': info.get('maxInputChannels', 0),
                            'sample_rate': info.get('defaultSampleRate', 0)
                        })
                p.terminate()
            except ImportError:
                pass

        return devices

    def _try_import_sounddevice(self) -> Any | None:
        """Try to import sounddevice."""
        try:
            import sounddevice as sd
            return sd
        except ImportError:
            return None

    def _try_import_pyaudio(self) -> Any | None:
        """Try to import pyaudio."""
        try:
            import pyaudio
            return pyaudio
        except ImportError:
            return None

    def _start_sounddevice(self) -> bool:
        """Start capture using sounddevice."""
        import sounddevice as sd

        def audio_callback(indata, frames, time_info, status):
            """Callback for sounddevice input stream."""
            if status:
                logger.warning(f"Sounddevice callback status: {status}")

            # Convert to numpy array and flatten
            audio_data = indata[:, 0] if len(indata.shape) > 1 else indata

            # Process frame
            self._process_audio_frame(audio_data.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                blocksize=self.config.chunk_size,
                device=self.config.device_index,
                callback=audio_callback
            )

            self._stream.start()
            self.state = CaptureState.RUNNING
            logger.info(f"Audio capture started with sounddevice (sr={self.config.sample_rate})")
            return True

        except Exception as e:
            raise AudioCaptureError(f"Failed to start sounddevice stream: {e}")

    def _start_pyaudio(self) -> bool:
        """Start capture using pyaudio (with background thread)."""
        import pyaudio

        p = pyaudio.PyAudio()

        try:
            device_index = self.config.device_index
            if device_index is None:
                device_index = p.get_default_input_device_info()['index']

            self._stream = p.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._pyaudio_callback
            )

            self._pyaudio = p
            self._stream.start_stream()
            self.state = CaptureState.RUNNING
            logger.info(f"Audio capture started with pyaudio (sr={self.config.sample_rate})")
            return True

        except Exception as e:
            p.terminate()
            raise AudioCaptureError(f"Failed to start pyaudio stream: {e}")

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """Callback for pyaudio input stream."""
        import pyaudio

        if status:
            logger.warning(f"PyAudio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Process frame
        self._process_audio_frame(audio_data)

        return (None, pyaudio.paContinue)

    def _process_audio_frame(self, audio_data: np.ndarray) -> None:
        """
        Process a single audio frame.

        Args:
            audio_data: Raw audio samples (float32, normalized -1 to 1)
        """
        # Apply noise gate
        gated_audio = self._apply_noise_gate(audio_data)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(gated_audio ** 2))

        # VAD processing
        is_speech = False
        vad_confidence = 0.0

        if self._vad_enabled and self._vad:
            result = self._vad.process_frame(gated_audio)
            is_speech = result.is_speech
            vad_confidence = result.confidence
        else:
            # Simple energy-based VAD
            is_speech = rms > self.config.noise_gate_threshold
            vad_confidence = min(rms * 10, 1.0)

        # Create frame
        frame = AudioFrame(
            data=gated_audio,
            timestamp=time.time(),
            is_speech=is_speech
        )

        # Queue frame
        if not self._frame_queue.full():
            self._frame_queue.put(frame)

        # Call frame callback
        if self.on_frame_callback:
            try:
                self.on_frame_callback(frame)
            except Exception as e:
                logger.warning(f"Frame callback error: {e}")

        # Speech segment detection
        self._detect_speech_segment(frame, vad_confidence)

    def _apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply noise gate to audio data.

        Args:
            audio_data: Input audio samples

        Returns:
            Gated audio samples
        """
        threshold = self.config.noise_gate_threshold
        attack_coeff = np.exp(-1.0 / (self.config.noise_gate_attack_ms * self.config.sample_rate / 1000))
        release_coeff = np.exp(-1.0 / (self.config.noise_gate_release_ms * self.config.sample_rate / 1000))

        result = np.zeros_like(audio_data)

        for i, sample in enumerate(audio_data):
            # Calculate envelope
            envelope = abs(sample)

            # Gate logic
            if self._gate_open:
                if envelope < threshold:
                    self._gate_envelope *= release_coeff
                    if self._gate_envelope < threshold:
                        self._gate_open = False
                else:
                    self._gate_envelope = max(self._gate_envelope, envelope)
            else:
                if envelope > threshold:
                    self._gate_envelope = envelope
                    self._gate_open = True
                else:
                    self._gate_envelope *= attack_coeff

            # Apply gain
            gain = self._gate_envelope if self._gate_open else 0.01
            result[i] = sample * gain

        return result

    def _detect_speech_segment(self, frame: AudioFrame, confidence: float) -> None:
        """
        Detect and accumulate speech segments.

        Args:
            frame: Current audio frame
            confidence: VAD confidence score
        """
        frame_duration_ms = frame.duration_ms
        padding_frames = self.config.vad_padding_ms / frame_duration_ms
        min_speech_frames = self.config.vad_min_speech_ms / frame_duration_ms

        if frame.is_speech:
            if not self._in_speech:
                # Start of speech
                self._in_speech = True
                self._speech_start_time = frame.timestamp
                self._speech_buffer = []
                self._speech_frames = 0

            self._speech_buffer.append(frame.data)
            self._speech_frames += 1
            self._silence_frames = 0

        else:
            if self._in_speech:
                # Continue to capture padding
                self._speech_buffer.append(frame.data)
                self._silence_frames += 1

                # End of speech detected
                if self._silence_frames >= padding_frames:
                    # Check minimum duration
                    if self._speech_frames >= min_speech_frames:
                        self._flush_speech_buffer()

                    self._in_speech = False
                    self._speech_buffer = []
                    self._speech_frames = 0

    def _flush_speech_buffer(self) -> None:
        """Flush accumulated speech buffer as a segment."""
        if not self._speech_buffer:
            return

        audio_data = np.concatenate(self._speech_buffer)

        segment = SpeechSegment(
            audio_data=audio_data,
            start_time=self._speech_start_time or time.time(),
            end_time=time.time(),
            confidence=0.8  # Default confidence
        )

        # Queue segment
        if not self._speech_queue.full():
            self._speech_queue.put(segment)

        # Call speech callback
        if self.on_speech_callback:
            try:
                self.on_speech_callback(segment)
            except Exception as e:
                logger.warning(f"Speech callback error: {e}")

        logger.debug(f"Speech segment detected: {segment.duration_ms:.0f}ms")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


def create_audio_capture(
    sample_rate: int = 16000,
    device_index: int | None = None,
    noise_gate_threshold: float = 0.01,
    vad_enabled: bool = True
) -> AudioCapture:
    """
    Create an AudioCapture instance with sensible defaults.

    Args:
        sample_rate: Audio sample rate (default 16kHz for speech)
        device_index: Audio device index (None for default)
        noise_gate_threshold: Noise gate threshold (0-1)
        vad_enabled: Enable voice activity detection

    Returns:
        Configured AudioCapture instance
    """
    config = AudioConfig(
        sample_rate=sample_rate,
        channels=1,
        chunk_size=512,
        device_index=device_index,
        noise_gate_threshold=noise_gate_threshold,
        vad_enabled=vad_enabled
    )

    return AudioCapture(config=config)
