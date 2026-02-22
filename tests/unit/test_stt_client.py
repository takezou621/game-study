"""Tests for STT (Speech-to-Text) client module."""

import asyncio
import io
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from audio.stt_client import (
    PartialTranscription,
    StreamingSTT,
    STTClient,
    STTClientError,
    STTConfig,
    STTLanguage,
    STTModel,
    TranscriptionResult,
    create_stt_client,
)
from src.exceptions import APIError, ConfigurationError, RateLimitError

try:
    from openai import AsyncOpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    OpenAIError = Exception


# ============================================================================
# STTModel Tests
# ============================================================================

class TestSTTModel:
    """Test STTModel enum."""

    def test_model_values(self):
        """Test all model values exist."""
        assert STTModel.WHISPER_1.value == "whisper-1"

    def test_model_from_string(self):
        """Test creating enum from string value."""
        model = STTModel("whisper-1")
        assert model == STTModel.WHISPER_1


# ============================================================================
# STTLanguage Tests
# ============================================================================

class TestSTTLanguage:
    """Test STTLanguage enum."""

    def test_language_values(self):
        """Test all language values exist."""
        assert STTLanguage.ENGLISH.value == "en"
        assert STTLanguage.JAPANESE.value == "ja"
        assert STTLanguage.SPANISH.value == "es"
        assert STTLanguage.FRENCH.value == "fr"
        assert STTLanguage.GERMAN.value == "de"
        assert STTLanguage.CHINESE.value == "zh"
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
        assert config.enable_vad_filter is True
        assert config.prompt is None

    def test_init_custom(self):
        """Test custom configuration values."""
        config = STTConfig(
            model=STTModel.WHISPER_1,
            language=STTLanguage.JAPANESE,
            temperature=0.3,
            enable_timestamps=True,
            enable_vad_filter=False,
            prompt="Game context"
        )
        assert config.language == STTLanguage.JAPANESE
        assert config.temperature == 0.3
        assert config.enable_timestamps is True
        assert config.enable_vad_filter is False
        assert config.prompt == "Game context"


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

    def test_default_values(self):
        """Test default values."""
        result = TranscriptionResult(
            text="Test",
            language="en",
            duration_ms=100.0
        )
        assert result.confidence == 0.0
        assert result.segments == []
        assert isinstance(result.timestamp, float)

    def test_timestamp_is_current(self):
        """Test timestamp is set to current time."""
        before = time.time()
        result = TranscriptionResult(
            text="Test",
            language="en",
            duration_ms=100.0
        )
        after = time.time()
        assert before <= result.timestamp <= after


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
        assert isinstance(partial.timestamp, float)

    def test_init_final(self):
        """Test final transcription."""
        partial = PartialTranscription(text="Hello world", is_final=True)
        assert partial.is_final is True


# ============================================================================
# STTClientError Tests
# ============================================================================

class TestSTTClientError:
    """Test STTClientError exception."""

    def test_error_creation(self):
        """Test creating STTClientError."""
        error = STTClientError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


# ============================================================================
# STTClient Initialization Tests
# ============================================================================

class TestSTTClientInit:
    """Test STTClient initialization."""

    def test_init_requires_api_key(self):
        """Test that initialization requires API key."""
        with pytest.raises(ConfigurationError) as exc_info:
            STTClient(api_key=None)
        assert "OpenAI API key is required" in str(exc_info.value)

    def test_init_empty_api_key(self):
        """Test that empty string API key is rejected."""
        with pytest.raises(ConfigurationError) as exc_info:
            STTClient(api_key="")
        assert "OpenAI API key is required" in str(exc_info.value)

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init_success(self):
        """Test successful initialization."""
        client = STTClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.enabled is True
        assert client._current_transcription is None
        assert isinstance(client.client, AsyncOpenAI)

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = STTConfig(
            language=STTLanguage.JAPANESE,
            temperature=0.5
        )
        client = STTClient(api_key="test-key", config=config)
        assert client.config.language == STTLanguage.JAPANESE
        assert client.config.temperature == 0.5

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init_default_config(self):
        """Test initialization uses default config when None provided."""
        client = STTClient(api_key="test-key", config=None)
        assert client.config.model == STTModel.WHISPER_1
        assert client.config.language == STTLanguage.ENGLISH

    def test_init_without_openai_package(self):
        """Test initialization fails when OpenAI package is not available."""
        with patch('audio.stt_client.OPENAI_AVAILABLE', False):
            with pytest.raises(ConfigurationError) as exc_info:
                STTClient(api_key="test-key")
            assert "OpenAI package is required" in str(exc_info.value)


# ============================================================================
# STTClient Transcribe Tests
# ============================================================================

class TestSTTClientTranscribe:
    """Test STTClient transcription methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_not_enabled(self):
        """Test transcription fails when not enabled."""
        client = STTClient(api_key="test-key")
        client.enabled = False

        with pytest.raises(STTClientError) as exc_info:
            await client.transcribe(b"fake audio data")
        assert "not enabled" in str(exc_info.value)

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
        assert result.segments == []

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

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_auto_language(self):
        """Test transcription with auto language detection."""
        client = STTClient(api_key="test-key", config=STTConfig(language=STTLanguage.AUTO))

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Auto detected text"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        # Check that language parameter was None (auto-detect)
        call_kwargs = client.client.audio.transcriptions.create.call_args[1]
        assert call_kwargs['language'] is None
        assert result.text == "Auto detected text"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_timestamps(self):
        """Test transcription with timestamps enabled."""
        config = STTConfig(enable_timestamps=True)
        client = STTClient(api_key="test-key", config=config)

        # Mock the API response with segments
        mock_response = MagicMock()
        mock_response.text = "Segmented text"
        mock_response.language = "en"

        # Create mock segments
        mock_segment = MagicMock()
        mock_segment.text = "Segmented text"
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_response.segments = [mock_segment]

        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        assert result.text == "Segmented text"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Segmented text"
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 1.5
        assert result.language == "en"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_timestamps_no_segments(self):
        """Test transcription with timestamps but no segments attribute."""
        config = STTConfig(enable_timestamps=True)
        client = STTClient(api_key="test-key", config=config)

        # Mock response without segments attribute
        mock_response = MagicMock()
        mock_response.text = "Text without segments"
        mock_response.language = "en"
        # Remove segments to test hasattr path
        del mock_response.segments

        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        assert result.text == "Text without segments"
        assert result.segments == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_string_response(self):
        """Test transcription when API returns string."""
        client = STTClient(api_key="test-key")

        # Mock string response (old API format)
        client.client.audio.transcriptions.create = AsyncMock(
            return_value="String response"
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        assert result.text == "String response"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_prompt(self):
        """Test transcription with custom prompt."""
        config = STTConfig(prompt="Game: Pokémon Battle")
        client = STTClient(api_key="test-key", config=config)

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Prompted transcription"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        # Verify prompt was passed
        call_kwargs = client.client.audio.transcriptions.create.call_args[1]
        assert call_kwargs['prompt'] == "Game: Pokémon Battle"
        assert result.text == "Prompted transcription"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_with_temperature(self):
        """Test transcription with custom temperature."""
        config = STTConfig(temperature=0.5)
        client = STTClient(api_key="test-key", config=config)

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Temperature test"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        # Verify temperature was passed
        call_kwargs = client.client.audio.transcriptions.create.call_args[1]
        assert call_kwargs['temperature'] == 0.5
        assert result.text == "Temperature test"


# ============================================================================
# STTClient Error Handling Tests
# ============================================================================

class TestSTTClientErrors:
    """Test STTClient error handling."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_openai_error(self):
        """Test transcription handles OpenAI API errors."""
        client = STTClient(api_key="test-key")

        # Mock API error
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("API request failed")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        with pytest.raises(APIError) as exc_info:
            await client.transcribe(audio_data)
        assert "STT transcription failed" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_rate_limit_error(self):
        """Test transcription handles rate limit errors."""
        client = STTClient(api_key="test-key")

        # Mock rate limit error with underscore (to trigger rate_limit detection)
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("rate_limit exceeded")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        with pytest.raises(RateLimitError) as exc_info:
            await client.transcribe(audio_data)
        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_rate_limit_error_capitalized(self):
        """Test transcription handles capitalized RATE_LIMIT error."""
        client = STTClient(api_key="test-key")

        # Mock rate limit error with underscore
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("RATE_LIMIT error occurred")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        with pytest.raises(RateLimitError) as exc_info:
            await client.transcribe(audio_data)
        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_generic_exception(self):
        """Test transcription handles generic exceptions."""
        client = STTClient(api_key="test-key")

        # Mock generic exception
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=ValueError("Unexpected error")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        with pytest.raises(STTClientError) as exc_info:
            await client.transcribe(audio_data)
        assert "Transcription error" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_close(self):
        """Test closing client."""
        client = STTClient(api_key="test-key")

        await client.close()

        assert client.enabled is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_tracks_duration(self):
        """Test transcription tracks processing duration."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Duration test"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)
        result = await client.transcribe(audio_data)

        assert result.duration_ms >= 0
        assert result.duration_ms < 1000  # Should be fast


# ============================================================================
# STTClient Stream Tests
# ============================================================================

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
        assert all(r.text == "Streaming test" for r in results)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_stream_with_bytes_chunks(self):
        """Test streaming with byte chunks."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Bytes streaming"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Create byte audio chunks
        # Need at least 16000 bytes for 1 second at 16kHz (int16)
        async def byte_stream():
            for _ in range(2):
                await asyncio.sleep(0.001)
                yield b"\x00\x01" * 8000  # 8000 int16 samples = 0.5 second

        results = []
        async for partial in client.transcribe_stream(byte_stream()):
            results.append(partial)

        # Should get one result after enough chunks accumulate (1 second threshold)
        # and potentially a final one
        assert len(results) >= 1
        assert all(r.text == "Bytes streaming" for r in results)

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

        # Callback should have been called once
        callback.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_stream_final_chunk(self):
        """Test streaming handles final incomplete chunk."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Final chunk"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Create stream with small final chunk
        async def audio_stream():
            # One full chunk (gets processed)
            yield (np.ones(16000) * 0.1).astype(np.float32)
            await asyncio.sleep(0.001)
            # Small final chunk (gets processed as final)
            yield (np.ones(8000) * 0.1).astype(np.float32)

        results = []
        async for partial in client.transcribe_stream(audio_stream()):
            results.append(partial)

        assert len(results) == 2
        # Last one should be marked as final
        assert results[-1].is_final is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_stream_final_chunk_with_callback(self):
        """Test streaming final chunk with on_partial callback."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Final with callback"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        callback = Mock()

        # Create stream with only a small chunk (will be final)
        async def audio_stream():
            # Small chunk (less than 1 second, will be final)
            yield (np.ones(8000) * 0.1).astype(np.float32)

        results = []
        async for partial in client.transcribe_stream(audio_stream(), on_partial=callback):
            results.append(partial)

        # Callback should be called for final chunk too
        assert callback.call_count >= 1
        assert results[-1].is_final is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_transcribe_stream_with_language_override(self):
        """Test streaming transcription with language override."""
        client = STTClient(api_key="test-key")

        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = "Japanese stream"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        async def audio_stream():
            await asyncio.sleep(0.001)
            yield (np.ones(16000) * 0.1).astype(np.float32)

        results = []
        async for partial in client.transcribe_stream(
            audio_stream(),
            language=STTLanguage.JAPANESE
        ):
            results.append(partial)

        assert len(results) == 1
        assert results[0].text == "Japanese stream"


# ============================================================================
# STTClient Audio Preparation Tests
# ============================================================================

class TestSTTClientPrepareAudio:
    """Test STTClient audio preparation methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_prepare_audio_file_path(self):
        """Test preparing audio from file path."""
        client = STTClient(api_key="test-key")

        # This should fail with FileNotFoundError for non-existent file
        with pytest.raises(FileNotFoundError):
            await client._prepare_audio_file("/nonexistent/file.wav")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_prepare_audio_bytes(self):
        """Test preparing audio from bytes."""
        client = STTClient(api_key="test-key")

        audio_bytes = b"\x00\x01" * 8000
        result = await client._prepare_audio_file(audio_bytes)

        # Result should be a BytesIO buffer (or BufferedReader)
        assert hasattr(result, 'read')

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_bytes_to_array(self):
        """Test bytes to array conversion."""
        client = STTClient(api_key="test-key")

        audio_bytes = b"\x00\x01" * 8000  # 16000 bytes = 8000 int16 samples
        array = client._bytes_to_array(audio_bytes)

        assert isinstance(array, np.ndarray)
        assert len(array) == 8000  # 8000 int16 samples

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_prepare_audio_numpy_int16(self):
        """Test preparing audio from int16 numpy array."""
        client = STTClient(api_key="test-key")

        audio_array = (np.ones(16000) * 100).astype(np.int16)
        result = await client._prepare_audio_file(audio_array)

        assert hasattr(result, 'read')

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_prepare_audio_numpy_float32(self):
        """Test preparing audio from float32 numpy array."""
        client = STTClient(api_key="test-key")

        audio_array = (np.ones(16000) * 0.5).astype(np.float32)
        result = await client._prepare_audio_file(audio_array)

        assert hasattr(result, 'read')

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_bytes_to_array_custom_sample_rate(self):
        """Test bytes to array with custom sample rate."""
        client = STTClient(api_key="test-key")

        audio_bytes = b"\x00\x01" * 4000
        array = client._bytes_to_array(audio_bytes, sample_rate=8000)

        assert len(array) == 4000


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

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_create_with_spanish(self):
        """Test creating client with Spanish language."""
        client = create_stt_client(
            api_key="test-key",
            language=STTLanguage.SPANISH
        )

        assert client.config.language == STTLanguage.SPANISH


# ============================================================================
# StreamingSTT Tests
# ============================================================================

class TestStreamingSTTInit:
    """Test StreamingSTT initialization."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init_default(self):
        """Test initialization with default parameters."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        assert streaming_stt.stt_client == stt_client
        assert streaming_stt.min_speech_ms == 500
        assert streaming_stt.silence_padding_ms == 500
        assert streaming_stt.vad is None
        assert streaming_stt._in_speech is False
        assert streaming_stt._speech_buffer == []
        assert streaming_stt._silence_frames == 0

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init_with_vad(self):
        """Test initialization with VAD detector."""
        stt_client = STTClient(api_key="test-key")
        mock_vad = MagicMock()

        streaming_stt = StreamingSTT(
            stt_client,
            vad_detector=mock_vad,
            min_speech_ms=1000,
            silence_padding_ms=300
        )

        assert streaming_stt.vad == mock_vad
        assert streaming_stt.min_speech_ms == 1000
        assert streaming_stt.silence_padding_ms == 300

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_init(self):
        """Test initialization."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        assert streaming_stt.stt_client == stt_client
        assert streaming_stt.min_speech_ms == 500


class TestStreamingSTTCallbacks:
    """Test StreamingSTT callback methods."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_on_transcription_callback(self):
        """Test setting transcription callback."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        callback = Mock()
        streaming_stt.on_transcription(callback)

        assert streaming_stt._on_transcription == callback

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_on_partial_callback(self):
        """Test setting partial transcription callback."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        callback = Mock()
        streaming_stt.on_partial(callback)

        assert streaming_stt._on_partial == callback


class TestStreamingSTTProcess:
    """Test StreamingSTT processing methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame_without_vad(self):
        """Test processing a frame without VAD."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Frame test"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        streaming_stt = StreamingSTT(stt_client, vad_detector=None)

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)
        result = await streaming_stt.process_frame(audio_frame)

        # Result is None when no VAD (no speech detected)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame_with_vad_speech(self):
        """Test processing a speech frame with VAD."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Speech detected"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Mock VAD that detects speech
        mock_vad_result = MagicMock()
        mock_vad_result.is_speech = True

        streaming_stt = StreamingSTT(
            stt_client,
            min_speech_ms=100,  # Low threshold
            silence_padding_ms=50
        )

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)

        # Process speech frame when not in speech state (transition)
        assert streaming_stt._in_speech is False
        streaming_stt._speech_start_time = None

        result = await streaming_stt.process_frame(audio_frame, vad_result=mock_vad_result)

        # Should buffer speech but not transcribe yet
        assert result is None
        assert streaming_stt._in_speech is True
        assert streaming_stt._speech_start_time is not None
        assert len(streaming_stt._speech_buffer) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame_continue_speech(self):
        """Test processing continues speech when already in speech state."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Continued speech"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Mock VAD that detects speech
        mock_vad_result = MagicMock()
        mock_vad_result.is_speech = True

        streaming_stt = StreamingSTT(
            stt_client,
            min_speech_ms=100,
            silence_padding_ms=50
        )

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)

        # Start speech
        await streaming_stt.process_frame(audio_frame, vad_result=mock_vad_result)
        assert streaming_stt._in_speech is True

        # Continue speech - should not reset start time or buffer
        original_start_time = streaming_stt._speech_start_time
        original_buffer_len = len(streaming_stt._speech_buffer)

        result = await streaming_stt.process_frame(audio_frame, vad_result=mock_vad_result)

        assert result is None
        assert streaming_stt._in_speech is True
        assert streaming_stt._speech_start_time == original_start_time
        assert len(streaming_stt._speech_buffer) == original_buffer_len + 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame_speech_to_silence(self):
        """Test processing speech followed by silence."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Complete speech"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        streaming_stt = StreamingSTT(
            stt_client,
            min_speech_ms=100,
            silence_padding_ms=100  # Need enough silence frames
        )

        # Create speech VAD result
        speech_vad = MagicMock()
        speech_vad.is_speech = True

        # Create silence VAD result
        silence_vad = MagicMock()
        silence_vad.is_speech = False

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)

        # Process speech frame
        await streaming_stt.process_frame(audio_frame, vad_result=speech_vad)
        assert streaming_stt._in_speech is True

        # Process multiple silence frames to trigger end of speech
        frame_duration_ms = len(audio_frame) / 16.0
        silence_frames_needed = int(streaming_stt.silence_padding_ms / frame_duration_ms) + 1

        for _ in range(silence_frames_needed):
            result = await streaming_stt.process_frame(audio_frame, vad_result=silence_vad)

        # Eventually should get a transcription result
        assert streaming_stt._in_speech is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame_short_speech(self):
        """Test that short speech segments are skipped."""
        stt_client = STTClient(api_key="test-key")

        streaming_stt = StreamingSTT(
            stt_client,
            min_speech_ms=1000  # Require at least 1 second
        )

        # Create speech VAD result
        speech_vad = MagicMock()
        speech_vad.is_speech = True

        # Create silence VAD result
        silence_vad = MagicMock()
        silence_vad.is_speech = False

        # Short audio frame (only 100 samples ~ 6ms)
        short_frame = (np.ones(100) * 0.1).astype(np.float32)

        # Process short speech
        await streaming_stt.process_frame(short_frame, vad_result=speech_vad)

        # Process silence
        frame_duration_ms = len(short_frame) / 16.0
        silence_frames_needed = int(streaming_stt.silence_padding_ms / frame_duration_ms) + 1

        for _ in range(silence_frames_needed):
            result = await streaming_stt.process_frame(short_frame, vad_result=silence_vad)

        # Short speech should be skipped (no API call made)
        assert streaming_stt._in_speech is False
        assert streaming_stt._speech_buffer == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_process_frame_with_vad_detector(self):
        """Test that VAD detector is used when available."""
        stt_client = STTClient(api_key="test-key")

        # Mock VAD detector
        mock_vad = MagicMock()
        mock_vad.process_frame = Mock(return_value=MagicMock(is_speech=False))

        streaming_stt = StreamingSTT(stt_client, vad_detector=mock_vad)

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)

        # Process without passing vad_result - should use detector
        result = await streaming_stt.process_frame(audio_frame)

        # VAD should have been called
        mock_vad.process_frame.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_flush_speech_buffer_empty(self):
        """Test flushing empty speech buffer."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        result = await streaming_stt._flush_speech_buffer()

        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_flush_speech_buffer_with_transcription(self):
        """Test flushing speech buffer with transcription."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Flushed speech"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        streaming_stt = StreamingSTT(stt_client, min_speech_ms=100)

        # Add audio to buffer
        audio = (np.ones(16000) * 0.1).astype(np.float32)
        streaming_stt._speech_buffer = [audio]

        result = await streaming_stt._flush_speech_buffer()

        assert result is not None
        assert result.text == "Flushed speech"
        assert streaming_stt._speech_buffer == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_flush_speech_buffer_too_short(self):
        """Test that short speech is not transcribed."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client, min_speech_ms=1000)

        # Add short audio to buffer (less than 1000ms)
        audio = (np.ones(100) * 0.1).astype(np.float32)
        streaming_stt._speech_buffer = [audio]

        result = await streaming_stt._flush_speech_buffer()

        # Should return None for too short speech
        assert result is None
        assert streaming_stt._speech_buffer == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_flush_speech_buffer_with_callback(self):
        """Test that transcription callback is invoked."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Callback invoked"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        callback = Mock()
        streaming_stt = StreamingSTT(stt_client, min_speech_ms=100)
        streaming_stt.on_transcription(callback)

        # Add audio to buffer
        audio = (np.ones(16000) * 0.1).astype(np.float32)
        streaming_stt._speech_buffer = [audio]

        result = await streaming_stt._flush_speech_buffer()

        # Callback should have been called
        callback.assert_called_once()
        assert result.text == "Callback invoked"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_flush_speech_buffer_handles_errors(self):
        """Test that transcription errors are handled gracefully."""
        stt_client = STTClient(api_key="test-key")

        # Mock API error
        stt_client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("API failed")
        )

        streaming_stt = StreamingSTT(stt_client, min_speech_ms=100)

        # Add audio to buffer
        audio = (np.ones(16000) * 0.1).astype(np.float32)
        streaming_stt._speech_buffer = [audio]

        result = await streaming_stt._flush_speech_buffer()

        # Should return None on error
        assert result is None
        assert streaming_stt._speech_buffer == []


class TestStreamingSTTReset:
    """Test StreamingSTT reset method."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    def test_reset(self):
        """Test reset method clears all state."""
        stt_client = STTClient(api_key="test-key")
        streaming_stt = StreamingSTT(stt_client)

        # Set some state
        streaming_stt._in_speech = True
        streaming_stt._speech_buffer = [np.array([1, 2, 3])]
        streaming_stt._silence_frames = 5
        streaming_stt._speech_start_time = time.time()

        # Reset
        streaming_stt.reset()

        assert streaming_stt._in_speech is False
        assert streaming_stt._speech_buffer == []
        assert streaming_stt._silence_frames == 0
        assert streaming_stt._speech_start_time is None


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
        assert result.language == "en"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_streaming_stt_full_workflow(self):
        """Test complete streaming STT workflow."""
        stt_client = STTClient(api_key="test-key")

        # Mock API response
        mock_response = MagicMock()
        mock_response.text = "Full workflow"
        stt_client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Create VAD mock
        mock_vad = MagicMock()

        # Create streaming STT
        streaming_stt = StreamingSTT(
            stt_client,
            vad_detector=mock_vad,
            min_speech_ms=100,
            silence_padding_ms=50
        )

        # Set callback
        callback_results = []
        streaming_stt.on_transcription(lambda r: callback_results.append(r))

        # Create speech VAD result
        speech_vad = MagicMock()
        speech_vad.is_speech = True

        # Create silence VAD result
        silence_vad = MagicMock()
        silence_vad.is_speech = False

        audio_frame = (np.ones(16000) * 0.1).astype(np.float32)

        # Simulate speech segment
        await streaming_stt.process_frame(audio_frame, vad_result=speech_vad)

        # Trigger silence
        frame_duration_ms = len(audio_frame) / 16.0
        silence_frames_needed = int(streaming_stt.silence_padding_ms / frame_duration_ms) + 1

        for _ in range(silence_frames_needed):
            await streaming_stt.process_frame(audio_frame, vad_result=silence_vad)

        # Check state was reset
        assert streaming_stt._in_speech is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_multiple_languages(self):
        """Test transcription with multiple languages."""
        for lang in [STTLanguage.ENGLISH, STTLanguage.JAPANESE, STTLanguage.SPANISH]:
            config = STTConfig(language=lang)
            client = STTClient(api_key="test-key", config=config)

            # Mock response
            mock_response = MagicMock()
            mock_response.text = f"Text in {lang.value}"
            client.client.audio.transcriptions.create = AsyncMock(
                return_value=mock_response
            )

            audio_data = (np.ones(16000) * 0.1).astype(np.int16)
            result = await client.transcribe(audio_data)

            assert result.text == f"Text in {lang.value}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
    async def test_error_recovery(self):
        """Test that client can recover from errors."""
        client = STTClient(api_key="test-key")

        # First call fails
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=OpenAIError("API Error")
        )

        audio_data = (np.ones(16000) * 0.1).astype(np.int16)

        # Should raise error
        with pytest.raises(APIError):
            await client.transcribe(audio_data)

        # Reset mock for success
        mock_response = MagicMock()
        mock_response.text = "Success after error"
        client.client.audio.transcriptions.create = AsyncMock(
            return_value=mock_response
        )

        # Should succeed now
        result = await client.transcribe(audio_data)
        assert result.text == "Success after error"
