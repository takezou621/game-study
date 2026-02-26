"""Domain layer exceptions.

This module contains exception classes specific to the domain layer.
All domain exceptions inherit from DomainError for consistent error handling.

Exception Hierarchy:
    DomainError (base - inherits from GameStudyError)
    ├── InvalidValueError - Invalid value for domain value objects
    ├── StateTransitionError - Invalid state transitions
    ├── ValidationError - Domain validation failures
    ├── EntityNotFoundError - Entity not found
    ├── BusinessRuleViolationError - Business rule violations
    └── TriggerEvaluationError - Trigger evaluation failures
"""

from typing import Any

from src.exceptions import GameStudyError


class DomainError(GameStudyError):
    """Base exception for domain layer errors.

    Domain errors represent violations of business rules or domain invariants.
    These are distinct from application/infrastructure errors in that they
    represent problems with the business logic itself.

    Attributes:
        entity_type: Type of domain entity involved
        field_name: Name of the field that caused the error
        expected: Expected value or constraint
        actual: Actual value that caused the error
    """

    error_code: str = "DOM000"

    def __init__(
        self,
        message: str,
        entity_type: str | None = None,
        field_name: str | None = None,
        expected: Any = None,
        actual: Any = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize DomainError.

        Args:
            message: Human-readable error message
            entity_type: Type of domain entity (e.g., "GameState", "Player")
            field_name: Name of the field with the issue
            expected: Expected value or constraint description
            actual: Actual value that violated the constraint
            context: Additional context information
            cause: Original exception that caused this error
        """
        context = context or {}
        context.update(
            {
                "entity_type": entity_type,
                "field_name": field_name,
                "expected": str(expected) if expected is not None else None,
                "actual": str(actual) if actual is not None else None,
                "layer": "domain",
            }
        )

        super().__init__(message, context=context, cause=cause)
        self.entity_type = entity_type
        self.field_name = field_name
        self.expected = expected
        self.actual = actual


class InvalidValueError(DomainError):
    """Exception raised when a value violates domain constraints.

    This is raised when a value object receives an invalid value,
    such as HP outside the valid range or negative ammo.

    Example:
        >>> raise InvalidValueError(
        ...     "HP must be between 0 and 100",
        ...     entity_type="HP",
        ...     field_name="value",
        ...     expected="0-100",
        ...     actual=150
        ... )
    """

    error_code: str = "DOM001"

    def __init__(
        self,
        message: str,
        value_name: str | None = None,
        value: Any = None,
        constraints: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize InvalidValueError.

        Args:
            message: Human-readable error message
            value_name: Name of the invalid value
            value: The invalid value
            constraints: Dictionary of constraint descriptions (e.g., {"min": 0, "max": 100})
            **kwargs: Additional arguments passed to DomainError
        """
        context = kwargs.pop("context", None) or {}
        if constraints:
            context["constraints"] = constraints
        if value_name:
            kwargs.setdefault("field_name", value_name)
        if value is not None:
            kwargs.setdefault("actual", value)

        super().__init__(message, context=context, **kwargs)
        self.value_name = value_name
        self.value = value
        self.constraints = constraints


class StateTransitionError(DomainError):
    """Exception raised when an invalid state transition is attempted.

    This is raised when trying to change from one valid state to another
    in a way that violates the domain's state machine rules.

    Example:
        >>> raise StateTransitionError(
        ...     "Cannot transition from DEAD to ALIVE",
        ...     entity_type="Player",
        ...     from_state="DEAD",
        ...     to_state="ALIVE"
        ... )
    """

    error_code: str = "DOM002"

    def __init__(
        self,
        message: str,
        from_state: str | None = None,
        to_state: str | None = None,
        allowed_transitions: list[str] | None = None,
        **kwargs,
    ):
        """Initialize StateTransitionError.

        Args:
            message: Human-readable error message
            from_state: Current state
            to_state: Attempted target state
            allowed_transitions: List of valid target states from current state
            **kwargs: Additional arguments passed to DomainError
        """
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "from_state": from_state,
                "to_state": to_state,
                "allowed_transitions": allowed_transitions,
            }
        )

        super().__init__(message, context=context, **kwargs)
        self.from_state = from_state
        self.to_state = to_state
        self.allowed_transitions = allowed_transitions


class ValidationError(DomainError):
    """Exception raised when domain validation fails.

    This is a general validation error for domain-level validation
    that doesn't fit into more specific categories.

    Example:
        >>> raise ValidationError(
        ...     "Session phase must be one of: LOBBY, ACTIVE, ENDED",
        ...     entity_type="Session",
        ...     field_name="phase",
        ...     violations=["Invalid phase value: UNKNOWN"]
        ... )
    """

    error_code: str = "DOM003"

    def __init__(
        self,
        message: str,
        violations: list[str] | None = None,
        field_errors: dict[str, str] | None = None,
        **kwargs,
    ):
        """Initialize ValidationError.

        Args:
            message: Human-readable error message
            violations: List of validation violation descriptions
            field_errors: Dictionary mapping field names to error messages
            **kwargs: Additional arguments passed to DomainError
        """
        context = kwargs.pop("context", None) or {}
        if violations:
            context["violations"] = violations
        if field_errors:
            context["field_errors"] = field_errors

        super().__init__(message, context=context, **kwargs)
        self.violations = violations or []
        self.field_errors = field_errors or {}


class EntityNotFoundError(DomainError):
    """Exception raised when a domain entity cannot be found.

    Example:
        >>> raise EntityNotFoundError(
        ...     "Player session not found",
        ...     entity_type="Session",
        ...     entity_id="session-123"
        ... )
    """

    error_code: str = "DOM004"

    def __init__(
        self,
        message: str,
        entity_id: str | None = None,
        **kwargs,
    ):
        """Initialize EntityNotFoundError.

        Args:
            message: Human-readable error message
            entity_id: ID of the entity that was not found
            **kwargs: Additional arguments passed to DomainError
        """
        context = kwargs.pop("context", None) or {}
        context["entity_id"] = entity_id

        super().__init__(message, context=context, **kwargs)
        self.entity_id = entity_id


class BusinessRuleViolationError(DomainError):
    """Exception raised when a business rule is violated.

    This is for complex business rules that span multiple entities
    or involve more than simple value validation.

    Example:
        >>> raise BusinessRuleViolationError(
        ...     "Cannot start session with knocked player",
        ...     rule_name="session_start_preconditions",
        ...     details={"player_status": "KNOCKED"}
        ... )
    """

    error_code: str = "DOM005"

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        details: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize BusinessRuleViolationError.

        Args:
            message: Human-readable error message
            rule_name: Name of the violated business rule
            details: Additional details about the violation
            **kwargs: Additional arguments passed to DomainError
        """
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "rule_name": rule_name,
                "details": details,
            }
        )

        super().__init__(message, context=context, **kwargs)
        self.rule_name = rule_name
        self.details = details or {}


class TriggerEvaluationError(DomainError):
    """Raised when trigger evaluation fails.

    This exception is raised when there is an error during the evaluation
    of trigger conditions, such as invalid field paths or type mismatches.

    Example:
        >>> raise TriggerEvaluationError(
        ...     trigger_id="p0_low_hp",
        ...     reason="Field 'player.status.hp' not found in state",
        ...     details={"available_fields": ["player.status.shield"]}
        ... )
    """

    error_code: str = "DOM006"

    def __init__(
        self,
        trigger_id: str,
        reason: str,
        details: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize trigger evaluation error.

        Args:
            trigger_id: Trigger identifier
            reason: Reason for failure
            details: Optional additional details
            **kwargs: Additional arguments passed to DomainError
        """
        message = f"Failed to evaluate trigger '{trigger_id}': {reason}"
        context = kwargs.pop("context", None) or {}
        if details:
            context["details"] = details
        context["trigger_id"] = trigger_id

        super().__init__(message, context=context, **kwargs)
        self.trigger_id = trigger_id
        self.reason = reason


__all__ = [
    "DomainError",
    "InvalidValueError",
    "StateTransitionError",
    "ValidationError",
    "EntityNotFoundError",
    "BusinessRuleViolationError",
    "TriggerEvaluationError",
]
