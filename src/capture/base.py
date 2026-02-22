"""Base class for video capture sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from utils.time import get_timestamp_ms

if TYPE_CHECKING:
    from typing import Self


class BaseCapture(ABC):
    """Base class for video capture sources.

    This abstract base class defines the interface for all video capture
    implementations including video files, screen capture, and camera inputs.

    Attributes:
        is_opened: Whether the capture source is currently open.
        frame_count: Number of frames processed since opening.
        start_time_ms: Timestamp when the first frame was captured.
        fps: Calculated frames per second based on frame timing.
    """

    def __init__(self) -> None:
        """Initialize capture.

        Initializes capture with default state values. Subclasses should
        call this method via super().__init__() before setting up
        their specific capture resources.
        """
        self.is_opened = False
        self.frame_count = 0
        self.start_time_ms = 0
        self.fps = 0.0

    def __enter__(self) -> Self:
        """
        Context manager entry.

        Opens the capture source when entering a with block.

        Returns:
            Self reference for use in with statements.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """
        Context manager exit.

        Ensures the capture source is properly closed when exiting a with block.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self.close()

    @abstractmethod
    def open(self) -> None:
        """
        Open capture source.

        This abstract method must be implemented by subclasses to establish
        the connection to the video source (file, camera, screen, etc.).
        Subclasses should set self.is_opened to True on success.
        """
        pass

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """
        Read a frame from capture source.

        This abstract method must be implemented by subclasses to retrieve
        the next frame from the video source.

        Returns:
            Frame as numpy array with shape (height, width, channels) in BGR format,
            or None if reading failed or end of stream was reached.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Get capture metadata.

        This abstract method must be implemented by subclasses to return
        information about the video source.

        Returns:
            Dictionary with metadata fields which may include:
                - width (int): Frame width in pixels
                - height (int): Frame height in pixels
                - fps (float): Frames per second
                - frame_count (int): Total number of frames (for files)
                - codec (str): Video codec name (for files)
                - duration_ms (int): Duration in milliseconds (for files)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close capture source.

        This abstract method must be implemented by subclasses to release
        any resources held by the capture source (file handles, camera
        connections, etc.). Subclasses should set self.is_opened to False.
        """
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over frames.

        Provides an iterator interface for processing frames in a loop.
        Yields frames until read() returns None.

        Yields:
            numpy array representing each frame in BGR format.

        Example:
            >>> with VideoFileCapture("video.mp4") as capture:
            ...     for frame in capture:
            ...         process_frame(frame)
        """
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def _update_fps(self) -> None:
        """Calculate FPS based on frame timing.

        Updates the fps attribute based on the number of frames processed
        and elapsed time since the first frame. This should be called
        after each frame is read.
        """
        if self.frame_count == 0:
            self.start_time_ms = get_timestamp_ms()
        else:
            elapsed_ms = get_timestamp_ms() - self.start_time_ms
            if elapsed_ms > 0:
                self.fps = (self.frame_count / elapsed_ms) * 1000
        self.frame_count += 1
