"""Health-related value objects: HP and Shield."""

from dataclasses import dataclass
from typing import Any

from domain.exceptions import InvalidValueError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HP:
    """Health Points value object.

    Represents player health with validation for valid range (0-100).
    Immutable to ensure value semantics.
    """

    value: int
    max_value: int = 100

    def __post_init__(self) -> None:
        """Validate HP is within valid range."""
        if not 0 <= self.value <= self.max_value:
            logger.warning(
                "Invalid HP value: out of range",
                extra={"value": self.value, "max_value": self.max_value},
            )
            raise InvalidValueError(
                message=f"HP must be between 0 and {self.max_value}, got {self.value}",
                entity_type="HP",
                value_name="value",
                value=self.value,
                constraints={"min": 0, "max": self.max_value},
            )

    @property
    def is_low(self) -> bool:
        """Check if HP is considered low (< 50)."""
        return self.value < 50

    @property
    def is_critical(self) -> bool:
        """Check if HP is critical (< 25)."""
        return self.value < 25

    @property
    def percentage(self) -> float:
        """Get HP as percentage of max."""
        return self.value / self.max_value

    def with_value(self, new_value: int) -> "HP":
        """Create new HP with different value."""
        return HP(value=new_value, max_value=self.max_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "max_value": self.max_value,
        }


@dataclass(frozen=True)
class Shield:
    """Shield value object.

    Represents player shield with validation for valid range (0-100).
    Immutable to ensure value semantics.
    """

    value: int
    max_value: int = 100

    def __post_init__(self) -> None:
        """Validate Shield is within valid range."""
        if not 0 <= self.value <= self.max_value:
            logger.warning(
                "Invalid Shield value: out of range",
                extra={"value": self.value, "max_value": self.max_value},
            )
            raise InvalidValueError(
                message=f"Shield must be between 0 and {self.max_value}, got {self.value}",
                entity_type="Shield",
                value_name="value",
                value=self.value,
                constraints={"min": 0, "max": self.max_value},
            )

    @property
    def is_active(self) -> bool:
        """Check if shield is active (> 0)."""
        return self.value > 0

    @property
    def percentage(self) -> float:
        """Get Shield as percentage of max."""
        return self.value / self.max_value

    def with_value(self, new_value: int) -> "Shield":
        """Create new Shield with different value."""
        return Shield(value=new_value, max_value=self.max_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "max_value": self.max_value,
        }
