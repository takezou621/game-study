"""Capture port interface for video/screen capture."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CaptureMetadata:
    """Metadata about the capture source."""

    width: int
    height: int
    fps: float
    source_type: str  # "video", "screen", "webcam"
    source_name: str


@runtime_checkable
class CapturePort(Protocol):
    """Port (interface) for capture functionality.

    This defines the contract that capture implementations must follow.
    """

    def open(self) -> None:
        """Open the capture source.

        Raises:
            CaptureError: If source cannot be opened
        """
        ...

    def read(self) -> np.ndarray | None:
        """Read a frame from the capture source.

        Returns:
            Frame as numpy array (H, W, C) in BGR format, or None if exhausted

        Raises:
            CaptureError: If frame cannot be read
        """
        ...

    def close(self) -> None:
        """Close the capture source and release resources."""
        ...

    def get_metadata(self) -> CaptureMetadata:
        """Get metadata about the capture source.

        Returns:
            CaptureMetadata with source information
        """
        ...

    @property
    def is_open(self) -> bool:
        """Check if capture source is open."""
        ...


class SyncCapturePort(Protocol):
    """Synchronous capture port for non-async contexts."""

    def open(self) -> None: ...

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...

    def get_metadata(self) -> CaptureMetadata: ...

    @property
    def is_open(self) -> bool: ...


class AsyncCapturePort(Protocol):
    """Asynchronous capture port for async contexts."""

    async def open(self) -> None: ...

    async def read(self) -> np.ndarray | None: ...

    async def close(self) -> None: ...

    def get_metadata(self) -> CaptureMetadata: ...

    @property
    def is_open(self) -> bool: ...
