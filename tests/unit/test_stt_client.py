"""Tests for STT (Speech-to-Text) client module."""

import asyncio
import io
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from audio.stt_client import (
    STTConfig,
    STTClient,
    STTClientError,
    STTLanguage,
    STTModel,
    TranscriptionResult,
    PartialTranscription,
    StreamingSTT,
    create_stt_client,
)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# STTModel Tests
# ============================================================================

class TestSTTModel:
    """Test STTModel enum."""

    def test_model_values(self):
        """Test all model values exist."""
        assert STTModel.WHISPER_1.value == "whisper-1"


# ============================================================================
# STTLanguage Tests
# ============================================================================

class TestSTTLanguage:
    """Test STTLanguage enum."""

    def test_language_values(self):
        """Test all language values exist."""
        assert STTLanguage.ENGLISH.value == "en"
        assert STTLanguage.JAPANESE.value == "ja"
        assert STTLanguage.AUTO.value == "auto"


# ============================================================================
# STTConfig Tests
# ============================================================================

class TestSTTConfig:
    """Test STTConfig dataclass."""

    def test_init_default(self):
        """Test default configuration values."""
        config = STTConfig()
        assert config.model == STTModel.WHISPER_1
        assert config.language == STTLanguage.ENGLISH
        assert config.temperature == 0.0
        assert config.enable_timestamps is False

    def test_init_custom(self):
        """Test custom configuration values."""
        config = STTConfig(
            model=STTModel.WHISPER_1,
            language=STTLanguage.JAPANESE,
            temperature=0.3,
            enable_timestamps=True
        )
        assert config.language == STTLanguage.JAPANESE
        assert config.temperature == 0.3
        assert config.enable_timestamps is True


# ============================================================================
# TranscriptionResult Tests
# ============================================================================

class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration_ms=1500.0,
            confidence=0.95,
            segments=[{"start": 0.0, "end": 1.5, "text": "Hello world"}]
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Hello world"
        assert result_dict["language"] == "en"
        assert result_dict["duration_ms"] == 1500.0
        assert result_dict["confidence"] == 0.95
        assert len(result_dict["segments"]) == 1


# ============================================================================
# PartialTranscription Tests
# ============================================================================

class TestPartialTranscription:
    """Test PartialTranscription dataclass."""

    def test_init_default(self):
        """Test default values."""
        partial = PartialTranscription(text="Hello")
        assert partial.text == "Hello"
        assert partial.is_final is False

    def test_init_final(self):
        """Test final transcription."""
        partial = PartialTranscription(text="Hello world", is_final=True)
        assert partial.is_final is True


# ============================================================================
# STTClient Tests
# ============================================================================

class TestSTTClientInit:
    """Test STTClient initialization."""

    def test_init_requires_api_key(self):
        """Test that initialization requires API key."""
        with pytest.raises(Exception):  # ConfigurationError
            STTClient(api_key=None)

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init_success(self):
        """Test successful initialization."""
        client = STTClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.enabled is True


class TestSTTClientTranscribe:
    """Test STTClient transcription methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_not_enabled(self):
        """Test transcription fails when not enabled."""
        client = STTClient(api_key="test-key")
        client.enabled = False

        with pytest.raises(STTClientError):
            await client.transcribe(b"fake audio data")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_mock(self):
        """Test transcription with mocked API."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Create fake audio data
        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        result = await client.transcribe(audio_data)

        assert result.text == "Hello world"
        assert result.language == "en"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_numpy_array(self):
        """Test transcription with numpy array input."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Test transcription"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.random.randn(16000) * 0.1).astype(np.float32)
        result = await client.transcribe(audio_data)

        assert result.text == "Test transcription"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_bytes(self):
        """Test transcription with bytes input."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Bytes transcription"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_bytes = b"\x00\x01" * 8000  # Fake PCM16 data
        result = await client.transcribe(audio_bytes)

        assert result.text == "Bytes transcription"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_language(self):
        """Test transcription with language override."""
        client = STTClient(api_key="test-key", config=STTConfig(language=STTLanguage.ENGLISH))

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Japanese text"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data, language=STTLanguage.JAPANESE)

        assert result.text == "Japanese text"


class TestSTTClientStream:
    """Test STTClient streaming methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_stream(self):
        """Test streaming transcription."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Streaming test"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Create audio chunks
        async def audio_stream():
            for _ in range(3):
                await asyncio.sleep(0.001)
                yield (np.ones(16000) * 0.1).astype(np.float32)

        results = []
        async for partial in client.transcribe_stream(audio_stream()):
            results.append(partial)

        assert len(results) == 3
        assert all(isinstance(r, PartialTranscription) for r in results)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_stream_with_partial_callback(self):
        """Test streaming transcription with partial callback."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Callback test"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        callback = Mock()

        async def audio_stream():
            await asyncio.sleep(0.001)
            yield (np.ones(16000) * 0.1).astype(np.float32)

        results = []
        async for partial in client.transcribe_stream(audio_stream(), on_partial=callback):
            results.append(partial)

        assert len(results) == 1


class TestSTTClientPrepareAudio:
    """Test STTClient audio preparation methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_prepare_audio_file_path(self):
        """Test preparing audio from file path."""
        client = STTClient(api_key="test-key")

        # This would normally open a file, but we'll just check
        # that it handles the path correctly
        with pytest.raises(FileNotFoundError):
            await client._prepare_audio_file("/nonexistent/file.wav")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_prepare_audio_bytes(self):
        """Test preparing audio from bytes."""
        client = STTClient(api_key="test-key")

        audio_bytes = b"\x00\x01" * 8000
        result = await client._prepare_audio_file(audio_bytes)

        assert isinstance(result, io.BufferedReader)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_bytes_to_array(self):
        """Test bytes to array conversion."""
        client = STTClient(api_key="test-key")

        audio_bytes = b"\x00\x01" * 8000
        array = client._bytes_to_array(audio_bytes)

        assert isinstance(array, np.ndarray)
        assert len(array) == 16000  # 8000 * 2 bytes / 2 (int16 to float32 ratio)


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestCreateSTTClient:
    """Test create_stt_client helper function."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_create_with_defaults(self):
        """Test creating client with default parameters."""
        client = create_stt_client(api_key="test-key")

        assert client.config.model == STTModel.WHISPER_1
        assert client.config.language == STTLanguage.ENGLISH
        assert client.config.temperature == 0.0

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_create_with_custom_params(self):
        """Test creating client with custom parameters."""
        client = create_stt_client(
            api_key="test-key",
            language=STTLanguage.JAPANESE
        )

        assert client.config.language == STTLanguage.JAPANESE


# ============================================================================
# StreamingSTT Tests
# ============================================================================

class TestStreamingSTTInit:
    """Test StreamingSTT initialization."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init(self):
        """Test initialization."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        assert streaming_stt.stt_client == stt_client
        assert streaming_stt.min_speech_ms == 500


class TestStreamingSTTProcess:
    """Test StreamingSTT processing methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame(self):
        """Test processing a single frame."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Frame test"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        streaming_stt = StreamingSTT(stt_client)

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)
        result = await streaming_stt.process_frame(audio_frame)

        # Result is None unless speech segment completes
        assert result is None or isinstance(result, TranscriptionResult)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_callbacks(self):
        """Test callback functionality."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Callback test"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        transcription_callback = Mock()
        partial_callback = Mock()

        streaming_stt = StreamingSTT(stt_client)
        streaming_stt.on_transcription(transcription_callback)
        streaming_stt.on_partial(partial_callback)

        assert streaming_stt._on_transcription == transcription_callback
        assert streaming_stt._on_partial == partial_callback

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_reset(self):
        """Test reset method."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        # Set some state
        streaming_stt._in_speech = True
        streaming_stt._speech_buffer = [np.array([1, 2, 3])]

        # Reset
        streaming_stt.reset()

        assert streaming_stt._in_speech is False
        assert streaming_stt._speech_buffer == []


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestSTTClientErrors:
    """Test STTClient error handling."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_api_error(self):
        """Test transcription handles API errors."""
        from openai import OpenAIError

        client = STTClient(api_key="test-key")

        # Mock API error
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("API Error")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        with pytest.raises(Exception):  # APIError or STTClientError
            await client.transcribe(audio_data)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_rate_limit_error(self):
        """Test transcription handles rate limit errors."""
        from openai import OpenAIError
        from utils.exceptions import RateLimitError

        client = STTClient(api_key="test-key")

        # Mock rate limit error
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("Rate limit exceeded")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        with pytest.raises((RateLimitError, Exception)):
            await client.transcribe(audio_data)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_close(self):
        """Test closing client."""
        client = STTClient(api_key="test-key")

        await client.close()

        assert client.enabled is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestSTTIntegration:
    """Integration tests for STT functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_full_transcription_flow(self):
        """Test complete transcription flow."""
        client = STTClient(api_key="test-key")

        # Mock successful transcription
        mock_response = MagicMock()
        mock_response.text = "Complete flow test"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Create audio data
        audio_data = (np.random.randn(16000) * 0.1).astype(np.float32)

        # Transcribe
        result = await client.transcribe(audio_data)

        assert result.text == "Complete flow test"
        assert result.duration_ms >= 0
