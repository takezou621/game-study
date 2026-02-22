"""Video capture modules for game-study."""

from .base import BaseCapture
from .screen_capture import ScreenCapture
from .video_file import VideoFileCapture

__all__ = ['VideoFileCapture', 'ScreenCapture', 'BaseCapture']
