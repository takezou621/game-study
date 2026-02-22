"""Speech-to-Text (STT) client using OpenAI Whisper API.

This module provides speech recognition functionality using OpenAI's
Whisper API for transcribing audio input.
"""

import io
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    from openai import AsyncOpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    OpenAIError = Exception

from utils.exceptions import APIError, ConfigurationError, RateLimitError
from utils.logger import get_logger

logger = get_logger(__name__)


class STTModel(Enum):
    """Available STT models."""
    WHISPER_1 = "whisper-1"
    # Future models can be added here


class STTLanguage(Enum):
    """Supported languages for STT."""
    ENGLISH = "en"
    JAPANESE = "ja"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    AUTO = "auto"  # Auto-detect


@dataclass
class STTConfig:
    """STT configuration."""
    model: STTModel = STTModel.WHISPER_1
    language: STTLanguage = STTLanguage.ENGLISH
    prompt: str | None = None  # Optional prompt for context
    temperature: float = 0.0  # Sampling temperature (0-1)
    enable_timestamps: bool = False
    enable_vad_filter: bool = True  # Enable VAD filtering


@dataclass
class TranscriptionResult:
    """Result from speech transcription."""
    text: str
    language: str
    duration_ms: float
    confidence: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "segments": self.segments
        }


@dataclass
class PartialTranscription:
    """Partial transcription result during streaming."""
    text: str
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)


class STTClientError(Exception):
    """Base exception for STT client errors."""
    pass


class STTClient:
    """
    Speech-to-Text client using OpenAI Whisper API.

    Features:
    - Async/await support
    - Batch and streaming transcription
    - Multi-language support
    - Error handling and retries
    - Audio preprocessing
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: STTConfig | None = None
    ):
        """
        Initialize STT client.

        Args:
            api_key: OpenAI API key
            config: STT configuration
        """
        self.api_key = api_key
        self.config = config or STTConfig()

        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key is required for STT",
                config_key="OPENAI_API_KEY"
            )

        # Initialize OpenAI client
        if not OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.enabled = True

        # State
        self._current_transcription: str | None = None

    async def transcribe(
        self,
        audio: np.ndarray | bytes | str,
        language: STTLanguage | None = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (numpy array, bytes, or file path)
            language: Language override (None to use config)

        Returns:
            TranscriptionResult with transcribed text

        Raises:
            APIError: If transcription fails
            RateLimitError: If rate limited
        """
        if not self.enabled:
            raise STTClientError("STT client is not enabled")

        # Prepare audio file
        audio_file = await self._prepare_audio_file(audio)

        # Determine language
        lang = language or self.config.language
        language_param = None if lang == STTLanguage.AUTO else lang.value

        start_time = time.time()

        try:
            response = await self.client.audio.transcriptions.create(
                model=self.config.model.value,
                file=audio_file,
                language=language_param,
                prompt=self.config.prompt,
                temperature=self.config.temperature,
                timestamp_granularities=["segment"] if self.config.enable_timestamps else None,
                response_format="verbose_json" if self.config.enable_timestamps else "text"
            )

            duration_ms = (time.time() - start_time) * 1000

            # Handle response format
            if self.config.enable_timestamps:
                text = response.text
                segments = []
                if hasattr(response, 'segments'):
                    for seg in response.segments:
                        segments.append({
                            "text": seg.text,
                            "start": seg.start,
                            "end": seg.end
                        })
                detected_language = getattr(response, 'language', lang.value)
            else:
                text = response if isinstance(response, str) else response.text
                segments = []
                detected_language = lang.value if lang != STTLanguage.AUTO else "en"

            return TranscriptionResult(
                text=text.strip(),
                language=detected_language,
                duration_ms=duration_ms,
                segments=segments
            )

        except OpenAIError as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(
                    "STT rate limit exceeded",
                    cause=e
                )
            raise APIError(
                f"STT transcription failed: {e}",
                endpoint="audio.transcriptions",
                cause=e
            )

        except Exception as e:
            raise STTClientError(f"Transcription error: {e}")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray | bytes],
        language: STTLanguage | None = None,
        on_partial: Callable[[PartialTranscription], None] | None = None
    ) -> AsyncIterator[PartialTranscription]:
        """
        Transcribe streaming audio.

        Note: OpenAI Whisper doesn't support true streaming, so this
        accumulates audio chunks and transcribes in batches.

        Args:
            audio_stream: Async iterator of audio chunks
            language: Language override
            on_partial: Callback for partial results

        Yields:
            PartialTranscription results
        """
        buffer = []
        buffer_duration_ms = 0
        chunk_duration_ms = 1000  # Process 1-second chunks

        lang = language or self.config.language

        async for chunk in audio_stream:
            # Convert to numpy if needed
            if isinstance(chunk, bytes):
                audio_array = self._bytes_to_array(chunk)
            else:
                audio_array = chunk

            buffer.append(audio_array)
            chunk_ms = len(audio_array) / 16.0  # Assuming 16kHz
            buffer_duration_ms += chunk_ms

            # Transcribe when buffer is full
            if buffer_duration_ms >= chunk_duration_ms:
                combined = np.concatenate(buffer)
                result = await self.transcribe(combined, lang)

                partial = PartialTranscription(
                    text=result.text,
                    is_final=False
                )

                if on_partial:
                    on_partial(partial)

                yield partial

                buffer = []
                buffer_duration_ms = 0

        # Transcribe remaining buffer
        if buffer:
            combined = np.concatenate(buffer)
            result = await self.transcribe(combined, lang)

            partial = PartialTranscription(
                text=result.text,
                is_final=True
            )

            if on_partial:
                on_partial(partial)

            yield partial

    async def _prepare_audio_file(
        self,
        audio: np.ndarray | bytes | str
    ) -> io.BufferedReader:
        """
        Prepare audio file for API submission.

        Args:
            audio: Audio data

        Returns:
            File-like object ready for API
        """
        # If it's a file path, open it
        if isinstance(audio, str):
            return open(audio, 'rb')

        # Convert bytes to numpy if needed
        if isinstance(audio, bytes):
            audio_array = self._bytes_to_array(audio)
        else:
            audio_array = audio

        # Convert to WAV format
        import wave

        # Ensure float32 normalized
        if audio_array.dtype == np.int16:
            audio_int16 = audio_array
        else:
            audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_int16.tobytes())

        wav_buffer.seek(0)
        return wav_buffer

    @staticmethod
    def _bytes_to_array(data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Convert raw bytes to numpy array."""
        # Assume 16-bit PCM
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    async def close(self) -> None:
        """Close the STT client and cleanup resources."""
        await self.client.close()
        self.enabled = False


class StreamingSTT:
    """
    Streaming STT with VAD integration.

    This class combines VAD and STT for real-time speech recognition,
    automatically detecting speech segments and transcribing them.
    """

    def __init__(
        self,
        stt_client: STTClient,
        vad_detector: Any | None = None,
        min_speech_ms: int = 500,
        silence_padding_ms: int = 500
    ):
        """
        Initialize streaming STT.

        Args:
            stt_client: STT client instance
            vad_detector: Optional VAD detector
            min_speech_ms: Minimum speech duration to transcribe
            silence_padding_ms: Silence padding after speech
        """
        self.stt_client = stt_client
        self.vad = vad_detector
        self.min_speech_ms = min_speech_ms
        self.silence_padding_ms = silence_padding_ms

        # State
        self._in_speech = False
        self._speech_buffer: list[np.ndarray] = []
        self._silence_frames = 0
        self._speech_start_time: float | None = None

        # Callbacks
        self._on_transcription: Callable[[TranscriptionResult], None] | None = None
        self._on_partial: Callable[[PartialTranscription], None] | None = None

    def on_transcription(
        self,
        callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """Set transcription callback."""
        self._on_transcription = callback

    def on_partial(
        self,
        callback: Callable[[PartialTranscription], None]
    ) -> None:
        """Set partial transcription callback."""
        self._on_partial = callback

    async def process_frame(
        self,
        audio: np.ndarray,
        vad_result: Any | None = None
    ) -> TranscriptionResult | None:
        """
        Process a single audio frame.

        Args:
            audio: Audio frame
            vad_result: Optional pre-computed VAD result

        Returns:
            TranscriptionResult if speech segment completed
        """
        # Run VAD if not provided
        if vad_result is None and self.vad:
            vad_result = self.vad.process_frame(audio)

        is_speech = vad_result.is_speech if vad_result else False

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_time = time.time()
                self._speech_buffer = []
                self._silence_frames = 0

            self._speech_buffer.append(audio)

        else:
            if self._in_speech:
                self._silence_frames += 1

                # End of speech after silence padding
                frame_duration_ms = len(audio) / 16.0  # 16kHz
                silence_padding_frames = self.silence_padding_ms / frame_duration_ms

                if self._silence_frames >= silence_padding_frames:
                    result = await self._flush_speech_buffer()
                    self._in_speech = False
                    return result

        return None

    async def _flush_speech_buffer(self) -> TranscriptionResult | None:
        """Transcribe and flush speech buffer."""
        if not self._speech_buffer:
            return None

        # Check minimum duration
        total_samples = sum(len(frame) for frame in self._speech_buffer)
        duration_ms = total_samples / 16.0  # 16kHz

        if duration_ms < self.min_speech_ms:
            logger.debug(f"Speech too short ({duration_ms:.0f}ms), skipping")
            self._speech_buffer = []
            return None

        # Combine frames
        audio_combined = np.concatenate(self._speech_buffer)
        self._speech_buffer = []

        # Transcribe
        try:
            result = await self.stt_client.transcribe(audio_combined)

            if self._on_transcription:
                self._on_transcription(result)

            return result

        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return None

    def reset(self) -> None:
        """Reset streaming state."""
        self._in_speech = False
        self._speech_buffer = []
        self._silence_frames = 0
        self._speech_start_time = None


def create_stt_client(
    api_key: str | None = None,
    language: STTLanguage = STTLanguage.ENGLISH
) -> STTClient:
    """
    Create an STT client with sensible defaults.

    Args:
        api_key: OpenAI API key
        language: Recognition language

    Returns:
        Configured STTClient instance
    """
    config = STTConfig(
        model=STTModel.WHISPER_1,
        language=language,
        temperature=0.0
    )

    return STTClient(api_key=api_key, config=config)
