"""Position value object for spatial data."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Position:
    """2D Position value object.

    Represents a point in 2D space, typically normalized coordinates (0-1).
    Immutable to ensure value semantics.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate coordinates."""
        # Allow normalized coordinates (0-1) or pixel coordinates
        pass  # No strict validation as we support both coordinate systems

    @property
    def is_normalized(self) -> bool:
        """Check if coordinates are normalized (0-1 range)."""
        return 0 <= self.x <= 1 and 0 <= self.y <= 1

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
        }

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)
