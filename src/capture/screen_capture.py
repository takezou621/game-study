"""Real-time screen capture using MSS."""

import queue
import threading
import time
from collections.abc import Iterator

import numpy as np

try:
    import mss
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

from .base import BaseCapture


class ScreenCapture(BaseCapture):
    """
    Real-time screen capture with buffering and frame synchronization.

    Supports desktop screen capture with minimal latency.
    """

    def __init__(
        self,
        monitor: int = 1,
        monitor_id: int | None = None,
        target_fps: int = 60,
        buffer_size: int = 10,
        capture_region: tuple[int, int, int, int] | None = None,
        max_width: int | None = None
    ):
        """
        Initialize screen capture.

        Args:
            monitor: Monitor number (1 = primary)
            monitor_id: Alias for monitor (for compatibility)
            target_fps: Target FPS for capture
            buffer_size: Frame buffer size for smooth playback
            capture_region: Optional capture region (left, top, width, height)
            max_width: Optional maximum width to resize frames
        """
        # Support both monitor and monitor_id parameters
        if monitor_id is not None:
            monitor = monitor_id

        super().__init__()

        self.monitor = monitor
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.frame_time = 1.0 / target_fps
        self.max_width = max_width
        self.capture_region = capture_region

        # MSS screen capture instance (initialized in open())
        self.sct = None
        self.monitor_info = None

        # Frame buffer for smooth playback
        self.frame_buffer = queue.Queue(maxsize=buffer_size)

        # Threading control
        self._running = False
        self._capture_thread = None
        self._last_frame_time = 0.0

        # Statistics
        self._dropped_frames = 0
        self._start_time = 0.0

    def _get_monitor_info(self, monitor: int) -> dict:
        """
        Get monitor information.

        Args:
            monitor: Monitor number

        Returns:
            Monitor information dictionary
        """
        monitors = self.sct.monitors

        if monitor < 1 or monitor >= len(monitors):
            raise ValueError(f"Monitor {monitor} not found. Available: 1-{len(monitors)-1}")

        return monitors[monitor]

    def open(self) -> None:
        """Open/start screen capture."""
        if self._running:
            return

        # Initialize MSS (requires display)
        if not MSS_AVAILABLE:
            raise RuntimeError(
                "MSS library is not available or display is not accessible. "
                "This module requires a display server (X11/Wayland on Linux)."
            )

        if self.sct is None:
            self.sct = mss.mss()
            self.monitor_info = self._get_monitor_info(self.monitor)

            # Apply capture region if specified
            if self.capture_region:
                left, top, width, height = self.capture_region
                self.monitor_info["left"] = left
                self.monitor_info["top"] = top
                self.monitor_info["width"] = width
                self.monitor_info["height"] = height

        self.is_opened = True
        self._running = True
        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time = time.time()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._capture_thread.start()

    def close(self) -> None:
        """Close/stop screen capture."""
        if not self._running:
            return

        self._running = False
        self.is_opened = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        """Background capture loop."""
        while self._running:
            start_time = time.time()

            # Capture frame
            frame = self._capture_frame()

            if frame is not None:
                # Put frame in buffer (non-blocking)
                try:
                    self.frame_buffer.put(frame, block=False)
                except queue.Full:
                    # Buffer full, drop frame
                    self._dropped_frames += 1
            else:
                self._dropped_frames += 1

            # Maintain target FPS
            elapsed = time.time() - start_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)

    def _capture_frame(self) -> np.ndarray | None:
        """
        Capture single frame from screen.

        Returns:
            Frame as numpy array or None if capture failed
        """
        try:
            # Capture screen
            screenshot = self.sct.grab(self.monitor_info)

            # Convert BGRA to BGR
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize if max_width is specified
            if self.max_width and frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (self.max_width, new_height))

            return frame

        except Exception:
            # Capture failed
            return None

    def read(self) -> np.ndarray | None:
        """
        Read frame from buffer (blocking).

        Returns:
            Frame or None if stopped
        """
        if not self._running and self.frame_buffer.empty():
            return None

        try:
            frame = self.frame_buffer.get(timeout=1.0)
            self._update_fps()
            return frame
        except queue.Empty:
            return None

    def read_latest_frame(self) -> np.ndarray | None:
        """
        Read latest frame from buffer (non-blocking).

        Returns:
            Latest frame or None if buffer is empty
        """
        if self.frame_buffer.empty():
            return None

        # Get latest frame (drop old frames)
        frame = None
        while not self.frame_buffer.empty():
            frame = self.frame_buffer.get_nowait()

        return frame

    def get_metadata(self) -> dict:
        """
        Get capture metadata.

        Returns:
            Dictionary containing capture metadata
        """
        if self.monitor_info is None:
            # Not initialized yet
            return {
                "monitor": self.monitor,
                "target_fps": self.target_fps,
                "actual_fps": self.fps,
                "buffer_size": self.buffer_size,
            }

        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        total_frames = self.frame_count

        return {
            "monitor": self.monitor,
            "width": self.monitor_info["width"],
            "height": self.monitor_info["height"],
            "target_fps": self.target_fps,
            "actual_fps": self.fps,
            "frame_count": total_frames,
            "dropped_frames": self._dropped_frames,
            "buffer_size": self.buffer_size,
            "latency_ms": self._estimate_latency(),
        }

    def _estimate_latency(self) -> float:
        """
        Estimate capture latency in milliseconds.

        Returns:
            Estimated latency in ms
        """
        if self.frame_buffer.empty():
            return 0.0

        # Approximate latency based on buffer fill level
        buffer_fill = self.frame_buffer.qsize()
        latency_frames = buffer_fill / 2.0  # Average buffer position
        latency_ms = (latency_frames / self.target_fps) * 1000.0

        return latency_ms

    def set_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> None:
        """
        Set capture region (ROI).

        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height
        """
        if self.monitor_info:
            self.monitor_info["left"] = x
            self.monitor_info["top"] = y
            self.monitor_info["width"] = width
            self.monitor_info["height"] = height

    def reset_region(self) -> None:
        """Reset capture region to full screen."""
        if self.sct is None:
            return

        monitor = self._get_monitor_info(self.monitor)
        if self.monitor_info:
            self.monitor_info["top"] = monitor["top"]
            self.monitor_info["left"] = monitor["left"]
            self.monitor_info["width"] = monitor["width"]
            self.monitor_info["height"] = monitor["height"]

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate through frames.

        Yields:
            Frame array
        """
        while self._running or not self.frame_buffer.empty():
            frame = self.read()
            if frame is not None:
                yield frame
            elif not self._running:
                break


# Import cv2 for color conversion
import cv2
