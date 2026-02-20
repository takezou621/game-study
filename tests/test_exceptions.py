#!/usr/bin/env python3
"""Tests for custom exceptions."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.exceptions import (
    GameStudyError,
    VisionError,
    TriggerError,
    DialogueError,
    CaptureError,
    AudioError,
    APIError,
    ConfigurationError,
    RateLimitError,
)


def test_game_study_error_base():
    """Test base GameStudyError exception."""
    error = GameStudyError("Test error message")

    assert str(error) == "Test error message"
    assert error.message == "Test error message"
    assert error.details == {}


def test_game_study_error_with_details():
    """Test GameStudyError with details."""
    details = {"code": 500, "context": "test"}
    error = GameStudyError("Test error", details=details)

    assert error.details == details
    assert "Details:" in str(error)


def test_game_study_error_to_dict():
    """Test converting exception to dictionary."""
    error = GameStudyError("Test error", details={"code": 500})
    error_dict = error.to_dict()

    assert error_dict["error_type"] == "GameStudyError"
    assert error_dict["message"] == "Test error"
    assert error_dict["details"]["code"] == 500


def test_vision_error():
    """Test VisionError exception."""
    error = VisionError("Vision processing failed")

    assert isinstance(error, GameStudyError)
    assert "VisionError" in type(error).__name__


def test_trigger_error():
    """Test TriggerError exception."""
    error = TriggerError("Trigger processing failed")

    assert isinstance(error, GameStudyError)
    assert "TriggerError" in type(error).__name__


def test_dialogue_error():
    """Test DialogueError exception."""
    error = DialogueError("Dialogue processing failed")

    assert isinstance(error, GameStudyError)
    assert "DialogueError" in type(error).__name__


def test_capture_error():
    """Test CaptureError exception."""
    error = CaptureError("Capture failed")

    assert isinstance(error, GameStudyError)
    assert "CaptureError" in type(error).__name__


def test_audio_error():
    """Test AudioError exception."""
    error = AudioError("Audio processing failed")

    assert isinstance(error, GameStudyError)
    assert "AudioError" in type(error).__name__


def test_api_error():
    """Test APIError exception."""
    error = APIError("API call failed")

    assert isinstance(error, GameStudyError)
    assert "APIError" in type(error).__name__


def test_configuration_error():
    """Test ConfigurationError exception."""
    error = ConfigurationError("Invalid configuration")

    assert isinstance(error, GameStudyError)
    assert "ConfigurationError" in type(error).__name__


def test_rate_limit_error_basic():
    """Test basic RateLimitError."""
    error = RateLimitError("Rate limit exceeded")

    assert isinstance(error, APIError)
    assert isinstance(error, GameStudyError)
    assert error.retry_after is None


def test_rate_limit_error_with_retry_after():
    """Test RateLimitError with retry_after."""
    error = RateLimitError("Rate limit exceeded", retry_after=5.0)

    assert error.retry_after == 5.0
    assert "retry_after" in error.details
    assert error.details["retry_after"] == 5.0


def test_rate_limit_error_to_dict():
    """Test RateLimitError to_dict."""
    error = RateLimitError("Rate limit exceeded", retry_after=10.0)
    error_dict = error.to_dict()

    assert error_dict["error_type"] == "RateLimitError"
    assert error_dict["details"]["retry_after"] == 10.0


def test_exception_hierarchy():
    """Test that all custom exceptions inherit from GameStudyError."""
    exceptions = [
        VisionError("test"),
        TriggerError("test"),
        DialogueError("test"),
        CaptureError("test"),
        AudioError("test"),
        APIError("test"),
        ConfigurationError("test"),
        RateLimitError("test"),
    ]

    for exc in exceptions:
        assert isinstance(exc, GameStudyError)
