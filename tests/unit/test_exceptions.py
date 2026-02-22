"""Tests for custom exceptions."""

from pathlib import Path

import pytest

# Direct module import
SRC_PATH = Path(__file__).parent.parent.parent / "src"
EXCEPTIONS_PATH = SRC_PATH / "exceptions.py"

import importlib.util

spec = importlib.util.spec_from_file_location("exceptions", EXCEPTIONS_PATH)
exceptions_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exceptions_module)


class TestGameStudyError:
    """Tests for base GameStudyError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = exceptions_module.GameStudyError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.cause is None

    def test_error_with_cause(self):
        """Test error with cause."""
        original = ValueError("Original error")
        error = exceptions_module.GameStudyError("Wrapper error", cause=original)
        assert "Wrapper error" in str(error)
        assert "Original error" in str(error)
        assert error.cause is original


class TestVisionErrors:
    """Tests for vision-related errors."""

    def test_ocr_error(self):
        """Test OCRError."""
        error = exceptions_module.OCRError("OCR failed")
        assert isinstance(error, exceptions_module.VisionError)
        assert isinstance(error, exceptions_module.GameStudyError)

    def test_detection_error(self):
        """Test DetectionError."""
        error = exceptions_module.DetectionError("Detection failed")
        assert isinstance(error, exceptions_module.VisionError)

    def test_roi_extraction_error(self):
        """Test ROIExtractionError."""
        error = exceptions_module.ROIExtractionError("ROI extraction failed")
        assert isinstance(error, exceptions_module.VisionError)


class TestTriggerErrors:
    """Tests for trigger-related errors."""

    def test_trigger_config_error(self):
        """Test TriggerConfigError."""
        error = exceptions_module.TriggerConfigError("Invalid config")
        assert isinstance(error, exceptions_module.TriggerError)

    def test_trigger_evaluation_error(self):
        """Test TriggerEvaluationError."""
        error = exceptions_module.TriggerEvaluationError("Evaluation failed")
        assert isinstance(error, exceptions_module.TriggerError)


class TestDialogueErrors:
    """Tests for dialogue-related errors."""

    def test_api_error(self):
        """Test APIError."""
        error = exceptions_module.APIError("API call failed")
        assert isinstance(error, exceptions_module.DialogueError)
        assert error.retry_after is None

    def test_api_error_with_retry_after(self):
        """Test APIError with retry_after."""
        error = exceptions_module.APIError("Rate limited", retry_after=30.0)
        assert error.retry_after == 30.0

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceeded."""
        error = exceptions_module.RateLimitExceeded()
        assert isinstance(error, exceptions_module.APIError)
        assert error.retry_after == 60.0
        assert "Rate limit" in str(error)

    def test_openai_error(self):
        """Test OpenAIError."""
        error = exceptions_module.OpenAIError("OpenAI API error")
        assert isinstance(error, exceptions_module.APIError)

    def test_tts_error(self):
        """Test TTSError."""
        error = exceptions_module.TTSError("TTS synthesis failed")
        assert isinstance(error, exceptions_module.DialogueError)


class TestCaptureErrors:
    """Tests for capture-related errors."""

    def test_video_capture_error(self):
        """Test VideoCaptureError."""
        error = exceptions_module.VideoCaptureError("Video capture failed")
        assert isinstance(error, exceptions_module.CaptureError)

    def test_screen_capture_error(self):
        """Test ScreenCaptureError."""
        error = exceptions_module.ScreenCaptureError("Screen capture failed")
        assert isinstance(error, exceptions_module.CaptureError)


class TestWebRTCErrors:
    """Tests for WebRTC-related errors."""

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = exceptions_module.AuthenticationError("Auth failed")
        assert isinstance(error, exceptions_module.WebRTCError)

    def test_connection_error(self):
        """Test WebRTC ConnectionError."""
        error = exceptions_module.ConnectionError("Connection failed")
        assert isinstance(error, exceptions_module.WebRTCError)


class TestConfigErrors:
    """Tests for configuration-related errors."""

    def test_file_not_found_error(self):
        """Test FileNotFoundError."""
        error = exceptions_module.FileNotFoundError("Config file not found")
        assert isinstance(error, exceptions_module.ConfigError)

    def test_invalid_config_error(self):
        """Test InvalidConfigError."""
        error = exceptions_module.InvalidConfigError("Invalid value")
        assert isinstance(error, exceptions_module.ConfigError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_inherit_from_base(self):
        """Test that all custom exceptions inherit from GameStudyError."""
        exception_classes = [
            exceptions_module.VisionError,
            exceptions_module.OCRError,
            exceptions_module.DetectionError,
            exceptions_module.TriggerError,
            exceptions_module.DialogueError,
            exceptions_module.APIError,
            exceptions_module.RateLimitExceeded,
            exceptions_module.CaptureError,
            exceptions_module.WebRTCError,
            exceptions_module.ConfigError,
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, exceptions_module.GameStudyError)

    def test_catch_with_base_class(self):
        """Test that exceptions can be caught with base class."""
        errors_to_test = [
            exceptions_module.OCRError("test"),
            exceptions_module.TriggerError("test"),
            exceptions_module.APIError("test"),
            exceptions_module.WebRTCError("test"),
        ]

        for error in errors_to_test:
            with pytest.raises(exceptions_module.GameStudyError):
                raise error
