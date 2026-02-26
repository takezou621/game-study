"""Video file capture implementation."""

from typing import Any

import numpy as np

from infrastructure.capture.base import BaseCapture, CaptureMetadata
from infrastructure.exceptions import CaptureError
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoFileCapture(BaseCapture):
    """Capture from video file."""

    def __init__(self, video_path: str, loop: bool = False):
        """
        Initialize video file capture.

        Args:
            video_path: Path to video file
            loop: Whether to loop video when it ends
        """
        self.video_path = video_path
        self.loop = loop

        self._cap: Any | None = None
        self._frame_count = 0
        self._is_open = False

    def open(self) -> None:
        """Open video file."""
        import cv2

        self._cap = cv2.VideoCapture(self.video_path)

        if not self._cap.isOpened():
            raise CaptureError(
                source=self.video_path,
                reason="Failed to open video file",
            )

        self._frame_count = 0
        self._is_open = True
        logger.info(f"Opened video file: {self.video_path}")

    def read(self) -> np.ndarray | None:
        """Read next frame."""
        if not self._is_open or self._cap is None:
            return None

        ret, frame = self._cap.read()

        if not ret:
            if self.loop:
                # Restart video
                self._cap.set(1, 0)  # CV_CAP_PROP_POS_FRAMES
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                return None

        self._frame_count += 1
        return frame

    def close(self) -> None:
        """Close video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info(f"Closed video file: {self.video_path}")

    def get_metadata(self) -> CaptureMetadata:
        """Get video metadata."""
        if self._cap is None:
            raise CaptureError(
                source=self.video_path,
                reason="Capture not open - call open() first",
            )

        return CaptureMetadata(
            width=int(self._cap.get(3)),  # CV_CAP_PROP_FRAME_WIDTH
            height=int(self._cap.get(4)),  # CV_CAP_PROP_FRAME_HEIGHT
            fps=self._cap.get(5),  # CV_CAP_PROP_FPS
            source_type="video",
            source_name=self.video_path,
        )

    @property
    def is_open(self) -> bool:
        """Check if video is open."""
        return self._is_open

    @property
    def frame_count(self) -> int:
        """Get number of frames read."""
        return self._frame_count
