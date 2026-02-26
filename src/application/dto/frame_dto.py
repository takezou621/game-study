"""Frame DTOs for frame processing pipeline."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.time import get_timestamp_ms


@dataclass
class FrameDTO:
    """Data transfer object for input frames."""

    frame: np.ndarray
    frame_number: int
    timestamp_ms: int = field(default_factory=get_timestamp_ms)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get frame shape (H, W, C)."""
        return self.frame.shape

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.frame.shape[1]

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.frame.shape[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without frame data)."""
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }


@dataclass
class DetectionResultDTO:
    """Result of a single detection."""

    field: str
    value: Any
    source: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


@dataclass
class FrameResultDTO:
    """Result of processing a frame."""

    frame_number: int
    timestamp_ms: int
    detections: list[DetectionResultDTO] = field(default_factory=list)
    state_changes: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    errors: list[str] = field(default_factory=list)

    def add_detection(self, detection: DetectionResultDTO) -> None:
        """Add a detection result."""
        self.detections.append(detection)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "detections": [d.to_dict() for d in self.detections],
            "state_changes": self.state_changes,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
        }
