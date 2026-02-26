"""Base capture interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CaptureMetadata:
    """Metadata about the capture source."""

    width: int
    height: int
    fps: float
    source_type: str  # "video", "screen", "webcam"
    source_name: str


class BaseCapture(ABC):
    """Abstract base class for capture implementations."""

    @abstractmethod
    def open(self) -> None:
        """Open the capture source."""
        ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Read a frame from the source."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the capture source."""
        ...

    @abstractmethod
    def get_metadata(self) -> CaptureMetadata:
        """Get metadata about the capture source."""
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Check if source is open."""
        ...

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
