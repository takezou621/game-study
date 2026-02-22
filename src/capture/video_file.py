"""Video file capture using OpenCV."""

from pathlib import Path

import cv2
import numpy as np

from .base import BaseCapture


class VideoFileCapture(BaseCapture):
    """Capture frames from video file using OpenCV."""

    def __init__(self, video_path: str):
        """
        Initialize video file capture.

        Args:
            video_path: Path to video file
        """
        super().__init__()
        self.video_path = Path(video_path)
        self.cap = None
        self.fps = 0
        self.width = 0
        self.height = 0
        # frame_count is already set by super().__init__()

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def open(self) -> None:
        """Open video file."""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_opened = True

    def close(self) -> None:
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.is_opened = False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read next frame from video.

        Returns:
            Tuple of (success, frame) where frame is None on failure
        """
        if self.cap is None:
            raise RuntimeError("Video file is not opened. Call open() first.")

        ret, frame = self.cap.read()
        if ret:
            self._update_fps()
        return ret, frame if ret else None

    def read(self) -> np.ndarray | None:
        """
        Read next frame from video.

        Returns:
            Frame as numpy array or None if failed or ended
        """
        ret, frame = self.read_frame()
        return frame if ret else None

    def get_frame_at(self, frame_index: int) -> np.ndarray | None:
        """
        Get frame at specific index.

        Args:
            frame_index: Frame index to retrieve

        Returns:
            Frame array or None if index is out of bounds
        """
        if self.cap is None:
            raise RuntimeError("Video file is not opened. Call open() first.")

        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_metadata(self) -> dict:
        """
        Get video metadata.

        Returns:
            Dictionary containing video metadata
        """
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_seconds": self.frame_count / self.fps if self.fps > 0 else 0,
        }

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __iter__(self):
        """
        Iterate through frames.

        Yields:
            Frame array
        """
        if self.cap is None:
            raise RuntimeError("Video file is not opened. Call open() first.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
