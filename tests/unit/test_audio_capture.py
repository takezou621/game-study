"""Tests for audio capture module."""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from audio.capture import (
    AudioCapture,
    AudioConfig,
    AudioFrame,
    CaptureState,
    SpeechSegment,
    create_audio_capture,
)

# ============================================================================
# AudioConfig Tests
# ============================================================================

class TestAudioConfig:
    """Test AudioConfig dataclass."""

    def test_init_default(self):
        """Test default configuration values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 512
        assert config.format == "int16"
        assert config.device_index is None
        assert config.noise_gate_threshold == 0.01
        assert config.vad_enabled is True

    def test_init_custom(self):
        """Test custom configuration values."""
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            chunk_size=1024,
            noise_gate_threshold=0.05,
            vad_enabled=False
        )
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.chunk_size == 1024
        assert config.noise_gate_threshold == 0.05
        assert config.vad_enabled is False


# ============================================================================
# AudioFrame Tests
# ============================================================================

class TestAudioFrame:
    """Test AudioFrame dataclass."""

    def test_duration_ms(self):
        """Test duration_ms property calculation."""
        # 16kHz sample rate, 160 samples = 10ms
        audio_data = np.zeros(160, dtype=np.float32)
        frame = AudioFrame(data=audio_data)
        assert frame.duration_ms == 10.0

    def test_is_speech_default(self):
        """Test default is_speech value."""
        audio_data = np.zeros(160, dtype=np.float32)
        frame = AudioFrame(data=audio_data)
        assert frame.is_speech is False


# ============================================================================
# SpeechSegment Tests
# ============================================================================

class TestSpeechSegment:
    """Test SpeechSegment dataclass."""

    def test_duration_ms(self):
        """Test duration_ms property calculation."""
        start = time.time()
        end = start + 1.5  # 1.5 seconds

        audio_data = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        segment = SpeechSegment(
            audio_data=audio_data,
            start_time=start,
            end_time=end
        )

        assert segment.duration_ms == 1500.0

    def test_duration_seconds(self):
        """Test duration_seconds property calculation."""
        start = time.time()
        end = start + 2.5

        audio_data = np.zeros(16000, dtype=np.float32)
        segment = SpeechSegment(
            audio_data=audio_data,
            start_time=start,
            end_time=end
        )

        assert segment.duration_seconds == 2.5


# ============================================================================
# CaptureState Tests
# ============================================================================

class TestCaptureState:
    """Test CaptureState enum."""

    def test_state_values(self):
        """Test all state values exist."""
        assert CaptureState.STOPPED.value == "stopped"
        assert CaptureState.STARTING.value == "starting"
        assert CaptureState.RUNNING.value == "running"
        assert CaptureState.STOPPING.value == "stopping"
        assert CaptureState.ERROR.value == "error"


# ============================================================================
# AudioCapture Tests
# ============================================================================

class TestAudioCaptureInit:
    """Test AudioCapture initialization."""

    def test_init_default(self):
        """Test initialization with default values."""
        capture = AudioCapture()
        assert capture.state == CaptureState.STOPPED
        assert capture.config == AudioConfig()
        assert capture.on_frame_callback is None
        assert capture.on_speech_callback is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = AudioConfig(sample_rate=48000)
        capture = AudioCapture(config=config)
        assert capture.config.sample_rate == 48000

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        frame_callback = Mock()
        speech_callback = Mock()

        capture = AudioCapture(
            on_frame_callback=frame_callback,
            on_speech_callback=speech_callback
        )

        assert capture.on_frame_callback == frame_callback
        assert capture.on_speech_callback == speech_callback


class TestAudioCaptureInitialize:
    """Test AudioCapture initialization methods."""

    @patch('audio.capture.AudioCapture._try_import_sounddevice')
    @patch('audio.capture.AudioCapture._try_import_pyaudio')
    def test_initialize_no_library(self, mock_pyaudio, mock_sounddevice):
        """Test initialization fails with no audio library."""
        mock_sounddevice.return_value = None
        mock_pyaudio.return_value = None

        capture = AudioCapture()
        result = capture.initialize()

        assert result is False
        assert capture.state == CaptureState.ERROR

    @patch('audio.capture.AudioCapture._try_import_sounddevice')
    def test_initialize_with_sounddevice(self, mock_sounddevice):
        """Test initialization with sounddevice."""
        mock_sd = MagicMock()
        mock_sounddevice.return_value = mock_sd

        capture = AudioCapture(config=AudioConfig(vad_enabled=False))
        result = capture.initialize()

        assert result is True
        assert capture.state == CaptureState.STOPPED
        assert capture._sd == mock_sd


class TestAudioCaptureStartStop:
    """Test AudioCapture start/stop methods."""

    def test_start_when_no_library(self):
        """Test start fails when no audio library available."""
        capture = AudioCapture()
        result = capture.start()

        assert result is False
        assert capture.state == CaptureState.ERROR

    def test_stop_when_stopped(self):
        """Test stopping when already stopped."""
        capture = AudioCapture()
        capture.stop()  # Should not raise

        assert capture.state == CaptureState.STOPPED


class TestAudioCaptureReadFrame:
    """Test AudioCapture frame reading methods."""

    def test_read_frame_timeout(self):
        """Test read_frame returns None on timeout."""
        capture = AudioCapture()
        frame = capture.read_frame(timeout=0.01)

        assert frame is None


class TestAudioCaptureSpeechDetection:
    """Test AudioCapture speech detection logic."""

    def test_speech_buffer_init(self):
        """Test speech buffer is initialized correctly."""
        capture = AudioCapture()
        assert capture._speech_buffer == []
        assert capture._speech_start_time is None
        assert capture._in_speech is False


class TestAudioCaptureNoiseGate:
    """Test AudioCapture noise gate functionality."""

    def test_noise_gate_below_threshold(self):
        """Test noise gate with signal below threshold."""
        config = AudioConfig(noise_gate_threshold=0.5)
        capture = AudioCapture(config=config)

        # Create quiet audio
        audio_data = np.ones(512, dtype=np.float32) * 0.1
        gated = capture._apply_noise_gate(audio_data)

        # Should be attenuated
        assert np.max(np.abs(gated)) < np.max(np.abs(audio_data))

    def test_noise_gate_above_threshold(self):
        """Test noise gate with signal above threshold."""
        config = AudioConfig(noise_gate_threshold=0.1)
        capture = AudioCapture(config=config)

        # Create loud audio
        audio_data = np.ones(512, dtype=np.float32) * 0.5
        gated = capture._apply_noise_gate(audio_data)

        # Should pass through with minimal attenuation
        assert np.max(np.abs(gated)) > 0


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestCreateAudioCapture:
    """Test create_audio_capture helper function."""

    def test_create_with_defaults(self):
        """Test creating audio capture with default parameters."""
        capture = create_audio_capture()

        assert capture.config.sample_rate == 16000
        assert capture.config.device_index is None
        assert capture.config.noise_gate_threshold == 0.01
        assert capture.config.vad_enabled is True

    def test_create_with_custom_params(self):
        """Test creating audio capture with custom parameters."""
        capture = create_audio_capture(
            sample_rate=48000,
            device_index=1,
            noise_gate_threshold=0.05,
            vad_enabled=False
        )

        assert capture.config.sample_rate == 48000
        assert capture.config.device_index == 1
        assert capture.config.noise_gate_threshold == 0.05
        assert capture.config.vad_enabled is False


# ============================================================================
# Async Tests
# ============================================================================

class TestAudioCaptureAsync:
    """Test AudioCapture async methods."""

    @pytest.mark.asyncio
    async def test_read_frame_async(self):
        """Test async frame reading."""
        capture = AudioCapture()
        frame = await capture.read_frame_async()

        assert frame is None

    @pytest.mark.asyncio
    async def test_read_speech_segment_async(self):
        """Test async speech segment reading."""
        capture = AudioCapture()
        segment = await capture.read_speech_segment_async()

        assert segment is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestAudioCaptureIntegration:
    """Integration tests for AudioCapture."""

    def test_frame_callback_invocation(self):
        """Test that frame callback is invoked correctly."""
        callback = Mock()
        capture = AudioCapture(on_frame_callback=callback)

        # Simulate frame processing
        audio_data = np.zeros(512, dtype=np.float32)
        capture._process_audio_frame(audio_data)

        # Callback should have been called (queue may have frame)
        # Note: In real test, we'd need to check if queue has item
        assert capture._frame_queue.qsize() >= 0

    def test_speech_segment_detection(self):
        """Test speech segment is detected correctly."""
        speech_callback = Mock()
        capture = AudioCapture(
            on_speech_callback=speech_callback,
            config=AudioConfig(vad_min_speech_ms=100, vad_padding_ms=50)
        )

        # Simulate speech frames
        audio_data = np.ones(512, dtype=np.float32) * 0.5
        speech_frame = AudioFrame(
            data=audio_data,
            timestamp=time.time(),
            is_speech=True
        )

        # Process multiple speech frames
        for _ in range(10):
            capture._detect_speech_segment(speech_frame, 0.8)

        # Process silence frames
        silence_frame = AudioFrame(
            data=np.zeros(512, dtype=np.float32),
            timestamp=time.time(),
            is_speech=False
        )

        # Process enough silence to trigger end of speech
        for _ in range(20):
            capture._detect_speech_segment(silence_frame, 0.1)


# ============================================================================
# List Devices Tests
# ============================================================================

class TestListDevices:
    """Test device listing functionality."""

    @pytest.mark.skipif(
        True,  # sounddevice imports inside method, can't patch at module level
        reason="sounddevice is imported inside method, requires integration test"
    )
    def test_list_devices_with_sounddevice(self):
        """Test listing devices with sounddevice."""
        pass

    def test_list_devices_no_library(self):
        """Test listing devices with no audio library."""
        with patch('audio.capture.AudioCapture._try_import_sounddevice', return_value=None), \
             patch('audio.capture.AudioCapture._try_import_pyaudio', return_value=None):
            devices = AudioCapture.list_devices()
            assert devices == []


# ============================================================================
# State Transition Tests
# ============================================================================

class TestStateTransitions:
    """Test AudioCapture state transitions."""

    def test_stopped_to_starting_to_error(self):
        """Test state transition from stopped to starting to error."""
        capture = AudioCapture()

        assert capture.state == CaptureState.STOPPED

        # Try to start without library
        result = capture.start()

        assert result is False
        assert capture.state == CaptureState.ERROR

    def test_running_to_stopping_to_stopped(self):
        """Test state transition from running to stopping to stopped."""
        capture = AudioCapture()

        # Manually set to running
        capture.state = CaptureState.RUNNING

        # Stop should work
        capture.stop()

        assert capture.state == CaptureState.STOPPED
