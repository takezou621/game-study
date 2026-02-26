"""Audio capture implementation."""

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np

from infrastructure.audio.capture.config import AudioConfig
from infrastructure.audio.processing.buffer import AudioFrame, SpeechBuffer, SpeechSegment
from infrastructure.audio.processing.noise_gate import NoiseGate
from infrastructure.exceptions import AudioError
from utils.logger import get_logger

logger = get_logger(__name__)


class CaptureState(Enum):
    """Audio capture state."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class AudioCapture:
    """
    Audio capture manager for microphone input.

    Handles audio device management, streaming capture, noise gating,
    and coordinates with VAD for speech detection.
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        on_frame_callback: Callable[[AudioFrame], None] | None = None,
        on_speech_callback: Callable[[SpeechSegment], None] | None = None,
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

        # Noise gate
        self._noise_gate = NoiseGate(
            threshold=self.config.noise_gate_threshold,
            attack_ms=self.config.noise_gate_attack_ms,
            release_ms=self.config.noise_gate_release_ms,
            sample_rate=self.config.sample_rate,
        )

        # Speech buffer
        self._speech_buffer = SpeechBuffer(
            padding_ms=self.config.vad_padding_ms,
            min_speech_ms=self.config.vad_min_speech_ms,
            max_speech_ms=self.config.vad_max_speech_ms,
        )

        # VAD
        self._vad: Any | None = None
        self._vad_enabled = self.config.vad_enabled

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
                raise AudioError(
                    operation="initialize",
                    reason="No audio library available. Install sounddevice or pyaudio.",
                )

            # Initialize VAD if enabled
            if self._vad_enabled:
                try:
                    from audio.vad import VoiceActivityDetector

                    self._vad = VoiceActivityDetector(sample_rate=self.config.sample_rate)
                    if not self._vad.initialize():
                        logger.warning("VAD initialization failed, continuing without VAD")
                        self._vad_enabled = False
                    else:
                        logger.info("VAD initialized successfully")
                except ImportError:
                    logger.warning("VAD module not available")
                    self._vad_enabled = False

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
                raise AudioError(
                    operation="start",
                    reason="No audio library available",
                )

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
        if self._speech_buffer.is_speech_active:
            segment = self._speech_buffer.flush()
            if segment and self.on_speech_callback:
                self.on_speech_callback(segment)

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
                callback=audio_callback,
            )

            self._stream.start()
            self.state = CaptureState.RUNNING
            logger.info(f"Audio capture started with sounddevice (sr={self.config.sample_rate})")
            return True

        except Exception as e:
            raise AudioError(
                operation="start_sounddevice",
                reason=f"Failed to start sounddevice stream: {e}",
                device_index=self.config.device_index,
                sample_rate=self.config.sample_rate,
            )

    def _start_pyaudio(self) -> bool:
        """Start capture using pyaudio (with background thread)."""
        import pyaudio

        p = pyaudio.PyAudio()

        try:
            device_index = self.config.device_index
            if device_index is None:
                device_index = p.get_default_input_device_info()["index"]

            self._stream = p.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._pyaudio_callback,
            )

            self._pyaudio = p
            self._stream.start_stream()
            self.state = CaptureState.RUNNING
            logger.info(f"Audio capture started with pyaudio (sr={self.config.sample_rate})")
            return True

        except Exception as e:
            p.terminate()
            raise AudioError(
                operation="start_pyaudio",
                reason=f"Failed to start pyaudio stream: {e}",
                device_index=self.config.device_index,
                sample_rate=self.config.sample_rate,
            )

    def _pyaudio_callback(self, in_data, _frame_count, _time_info, status):
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
        gated_audio = self._noise_gate.process(audio_data)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(gated_audio**2))

        # VAD processing
        is_speech = False

        if self._vad_enabled and self._vad:
            result = self._vad.process_frame(gated_audio)
            is_speech = result.is_speech
        else:
            # Simple energy-based VAD
            is_speech = rms > self.config.noise_gate_threshold

        # Create frame
        frame = AudioFrame(
            data=gated_audio,
            timestamp=time.time(),
            is_speech=is_speech,
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
        segment = self._speech_buffer.add_frame(gated_audio, is_speech)
        if segment:
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
    vad_enabled: bool = True,
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
        vad_enabled=vad_enabled,
    )

    return AudioCapture(config=config)
