"""Domain layer for game coach application.

This layer contains:
- Entities: Core domain objects with identity
- Value Objects: Immutable objects without identity
- Events: Domain events for communication
- Services: Domain services for business logic
- Triggers: Trigger rules and policies
- Exceptions: Domain-specific exceptions
"""

from domain.entities.game_state import GameState
from domain.entities.player import PlayerStatus
from domain.entities.session import Session
from domain.events.base import DomainEvent
from domain.exceptions import (
    BusinessRuleViolationError,
    DomainError,
    EntityNotFoundError,
    InvalidValueError,
    StateTransitionError,
    TriggerEvaluationError,
    ValidationError,
)
from domain.value_objects.ammo import Ammo
from domain.value_objects.health import HP, Shield

__all__ = [
    # Entities
    "GameState",
    "PlayerStatus",
    "Session",
    # Value Objects
    "HP",
    "Shield",
    "Ammo",
    # Events
    "DomainEvent",
    # Exceptions
    "DomainError",
    "InvalidValueError",
    "StateTransitionError",
    "ValidationError",
    "EntityNotFoundError",
    "BusinessRuleViolationError",
    "TriggerEvaluationError",
]
