"""Video capture modules for game-study."""

from .video_file import VideoFileCapture
from .screen_capture import ScreenCapture
from .base import BaseCapture

__all__ = ['VideoFileCapture', 'ScreenCapture', 'BaseCapture']
