"""Base class for video capture sources."""

import numpy as np
import time
from typing import Optional, Iterator
from pathlib import Path
from abc import ABC, abstractmethod
from utils.time import get_timestamp_ms


class BaseCapture(ABC):
    """Base class for video capture sources."""

    def __init__(self):
        """Initialize capture."""
        self.is_opened = False
        self.frame_count = 0
        self.start_time_ms = 0
        self.fps = 0.0

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @abstractmethod
    def open(self) -> None:
        """Open capture source."""
        pass

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from capture source.

        Returns:
            Frame as numpy array or None if failed
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Get capture metadata.

        Returns:
            Dictionary with metadata (width, height, fps, etc.)
        """
        pass

    @abstractmethod
    def close(self):
        """Close capture source."""
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def _update_fps(self):
        """Calculate FPS based on frame timing."""
        if self.frame_count == 0:
            self.start_time_ms = get_timestamp_ms()
        else:
            elapsed_ms = get_timestamp_ms() - self.start_time_ms
            if elapsed_ms > 0:
                self.fps = (self.frame_count / elapsed_ms) * 1000
        self.frame_count += 1
