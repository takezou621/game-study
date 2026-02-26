"""Capture infrastructure."""

from infrastructure.capture.base import BaseCapture
from infrastructure.capture.factory import CaptureFactory
from infrastructure.capture.screen_capture import ScreenCapture
from infrastructure.capture.video_file import VideoFileCapture

__all__ = [
    "BaseCapture",
    "VideoFileCapture",
    "ScreenCapture",
    "CaptureFactory",
]
