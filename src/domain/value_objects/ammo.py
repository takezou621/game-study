"""Ammo value object."""

from dataclasses import dataclass
from typing import Any

from domain.exceptions import InvalidValueError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Ammo:
    """Ammunition value object.

    Represents weapon ammunition count.
    Immutable to ensure value semantics.
    """

    value: int
    max_value: int | None = None

    def __post_init__(self) -> None:
        """Validate ammo is non-negative."""
        if self.value < 0:
            logger.warning(
                "Invalid ammo value: negative value provided",
                extra={"value": self.value, "max_value": self.max_value},
            )
            raise InvalidValueError(
                message=f"Ammo cannot be negative, got {self.value}",
                entity_type="Ammo",
                value_name="value",
                value=self.value,
                constraints={"min": 0, "max": self.max_value},
            )
        if self.max_value is not None and self.value > self.max_value:
            logger.warning(
                "Invalid ammo value: exceeds max",
                extra={"value": self.value, "max_value": self.max_value},
            )
            raise InvalidValueError(
                message=f"Ammo cannot exceed max ({self.max_value}), got {self.value}",
                entity_type="Ammo",
                value_name="value",
                value=self.value,
                constraints={"min": 0, "max": self.max_value},
            )

    @property
    def is_empty(self) -> bool:
        """Check if out of ammo."""
        return self.value == 0

    @property
    def is_low(self) -> bool:
        """Check if ammo is low (< 10 or < 20% of max)."""
        if self.max_value is None:
            return self.value < 10
        return self.value < self.max_value * 0.2

    @property
    def percentage(self) -> float | None:
        """Get ammo as percentage of max, or None if max unknown."""
        if self.max_value is None:
            return None
        return self.value / self.max_value

    def with_value(self, new_value: int) -> "Ammo":
        """Create new Ammo with different value."""
        return Ammo(value=new_value, max_value=self.max_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "max_value": self.max_value,
        }
