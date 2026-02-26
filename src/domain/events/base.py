"""Base domain event class."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from utils.time import get_timestamp_ms


@dataclass
class DomainEvent:
    """Base class for domain events.

    Domain events represent something that happened in the domain
    and are used for communication between bounded contexts.
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp_ms: int = field(default_factory=get_timestamp_ms)
    event_type: str = "domain_event"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp_ms": self.timestamp_ms,
            "event_type": self.event_type,
        }
