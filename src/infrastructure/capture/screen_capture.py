"""Screen capture implementation."""

from typing import Any

import numpy as np

from infrastructure.capture.base import BaseCapture, CaptureMetadata
from infrastructure.exceptions import CaptureError, DeviceNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


class ScreenCapture(BaseCapture):
    """Capture from screen."""

    def __init__(
        self,
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None,
    ):
        """
        Initialize screen capture.

        Args:
            monitor_index: Monitor to capture (1 = primary)
            region: Optional region (x, y, width, height)
        """
        self.monitor_index = monitor_index
        self.region = region

        self._sct: Any | None = None
        self._is_open = False
        self._frame_count = 0

    def open(self) -> None:
        """Initialize screen capture."""
        import mss

        self._sct = mss.mss()

        # Validate monitor
        monitors = self._sct.monitors
        if self.monitor_index >= len(monitors):
            raise DeviceNotFoundError(
                device_type="video",
                device_id=self.monitor_index,
                available_devices=list(range(len(monitors))),
            )

        self._is_open = True
        logger.info(f"Screen capture initialized for monitor {self.monitor_index}")

    def read(self) -> np.ndarray | None:
        """Capture screen frame."""
        if not self._is_open or self._sct is None:
            return None

        try:
            if self.region:
                monitor = {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2],
                    "height": self.region[3],
                }
            else:
                monitor = self._sct.monitors[self.monitor_index]

            screenshot = self._sct.grab(monitor)
            frame = np.array(screenshot)

            # Convert BGRA to BGR
            frame = frame[:, :, :3]

            self._frame_count += 1
            return frame

        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None

    def close(self) -> None:
        """Close screen capture."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        self._is_open = False
        logger.info("Screen capture closed")

    def get_metadata(self) -> CaptureMetadata:
        """Get capture metadata."""
        if self._sct is None:
            raise CaptureError(
                source=f"screen://monitor:{self.monitor_index}",
                reason="Capture not open - call open() first",
            )

        monitor = self._sct.monitors[self.monitor_index]

        return CaptureMetadata(
            width=monitor.get("width", 1920),
            height=monitor.get("height", 1080),
            fps=60.0,  # Screen capture is typically 60fps
            source_type="screen",
            source_name=f"Monitor {self.monitor_index}",
        )

    @property
    def is_open(self) -> bool:
        """Check if capture is open."""
        return self._is_open

    @property
    def frame_count(self) -> int:
        """Get number of frames captured."""
        return self._frame_count
