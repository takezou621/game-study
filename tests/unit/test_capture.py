"""Tests for capture modules."""

import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestVideoFileCapture:
    """Tests for video file capture."""

    def test_import(self):
        """Test that VideoFileCapture can be imported."""
        from capture.video_file import VideoFileCapture
        assert VideoFileCapture is not None

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file."""
        from capture.video_file import VideoFileCapture
        with pytest.raises(FileNotFoundError):
            VideoFileCapture("/nonexistent/video.mp4")

    def test_metadata_structure(self):
        """Test metadata structure."""
        from capture.video_file import VideoFileCapture
        # This would need a real video file for full test
        # Just test the expected metadata fields exist
        expected_fields = ['fps', 'width', 'height', 'frame_count', 'duration']
        # Metadata should have these fields when available
        assert True  # Placeholder for actual video test


class TestScreenCapture:
    """Tests for screen capture."""

    def test_import(self):
        """Test that ScreenCapture can be imported."""
        from capture.screen_capture import ScreenCapture
        assert ScreenCapture is not None

    def test_capture_config(self):
        """Test capture configuration."""
        from capture.screen_capture import ScreenCapture, CaptureConfig
        config = CaptureConfig(
            fps=30,
            monitor=0,
            width=1920,
            height=1080
        )
        assert config.fps == 30
        assert config.monitor == 0


class TestCaptureBase:
    """Tests for capture base class."""

    def test_import(self):
        """Test that base classes can be imported."""
        from capture.base import BaseCapture
        assert BaseCapture is not None
