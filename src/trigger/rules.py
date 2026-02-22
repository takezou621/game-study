"""Trigger rule definition and condition evaluation."""

from enum import Enum
from typing import Any

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create a minimal BaseModel fallback if pydantic is not available
    class BaseModel:
        """Fallback BaseModel for when pydantic is not available."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return self.__dict__

        @classmethod
        def model_validate(cls, data):
            return cls(**data)

    def Field(default=None, **kwargs):
        return default


class OperatorType(str, Enum):
    """Valid comparison operators for trigger conditions."""
    EQ = "eq"
    LT = "lt"
    GT = "gt"
    LTE = "lte"
    GTE = "gte"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


if PYDANTIC_AVAILABLE:
    class TriggerConditionModel(BaseModel):
        """Pydantic model for trigger condition validation."""

        field: str = Field(..., description="Field path (e.g., 'player.status.hp')")
        operator: OperatorType = Field(..., description="Comparison operator")
        value: Any = Field(..., description="Value to compare against")

        @field_validator('field')
        @classmethod
        def validate_field(cls, v):
            """Validate field path format."""
            if not v or not isinstance(v, str):
                raise ValueError("Field must be a non-empty string")
            if '.' not in v:
                raise ValueError("Field must be a dot-separated path (e.g., 'player.status.hp')")
            return v

        @field_validator('operator')
        @classmethod
        def validate_operator(cls, v):
            """Validate operator is supported."""
            valid_operators = {op.value for op in OperatorType}
            if v not in valid_operators:
                raise ValueError(f"Operator must be one of {valid_operators}")
            return v

        class Config:
            use_enum_values = True


    class TriggerRuleModel(BaseModel):
        """Pydantic model for trigger rule validation."""

        rule_id: str = Field(..., min_length=1, description="Unique rule identifier")
        name: str = Field(..., min_length=1, description="Human-readable name")
        priority: int = Field(..., ge=0, le=3, description="Priority level (0=highest, 3=lowest)")
        enabled: bool = Field(default=True, description="Whether the rule is enabled")
        conditions: list[TriggerConditionModel] = Field(
            default_factory=list,
            description="List of conditions that must all be true"
        )
        templates: dict[str, str | None] = Field(
            default_factory=dict,
            description="Response templates for different movement states"
        )
        cooldown_ms: int = Field(default=0, ge=0, description="Cooldown period in milliseconds")
        interrupt_higher_priority: bool = Field(
            default=False,
            description="Whether to interrupt higher priority triggers"
        )

        @field_validator('templates')
        @classmethod
        def validate_templates(cls, v):
            """Validate templates contain valid states."""
            valid_states = {'combat', 'non_combat', 'default'}
            for key in v:
                if key not in valid_states:
                    raise ValueError(f"Invalid template state: {key}. Must be one of {valid_states}")
            return v

        @field_validator('rule_id')
        @classmethod
        def validate_rule_id(cls, v):
            """Validate rule ID format."""
            if not v or not isinstance(v, str):
                raise ValueError("Rule ID must be a non-empty string")
            return v

        class Config:
            use_enum_values = True


class TriggerCondition:
    """Condition for trigger evaluation."""

    def __init__(self, field: str, operator: str, value: Any):
        """
        Initialize trigger condition.

        Args:
            field: Field path (e.g., "player.status.hp")
            operator: Comparison operator (eq, lt, gt, lte, gte, ne, in, not_in, contains)
            value: Value to compare against

        Raises:
            ValueError: If parameters are invalid (when using Pydantic validation)
        """
        # Validate using Pydantic model if available
        if PYDANTIC_AVAILABLE:
            validated = TriggerConditionModel(
                field=field,
                operator=operator,
                value=value
            )
            self.field = validated.field
            self.operator = validated.operator
            self.value = validated.value
        else:
            # Fallback to basic validation
            self.field = field
            self.operator = operator
            self.value = value

    def evaluate(self, state: dict[str, Any]) -> bool:
        """
        Evaluate condition against game state.

        Args:
            state: Game state dictionary

        Returns:
            True if condition is met, False otherwise
        """
        # Navigate to the field value
        keys = self.field.split('.')
        current = state

        try:
            for key in keys:
                if key in current:
                    if isinstance(current[key], dict) and "value" in current[key]:
                        current = current[key]["value"]
                    else:
                        current = current[key]
                else:
                    return False
        except (KeyError, TypeError):
            return False

        # If value is None, condition is not met
        if current is None:
            return False

        # Evaluate the condition
        if self.operator == "eq":
            return current == self.value
        elif self.operator == "lt":
            return current < self.value
        elif self.operator == "gt":
            return current > self.value
        elif self.operator == "lte":
            return current <= self.value
        elif self.operator == "gte":
            return current >= self.value
        elif self.operator == "ne":
            return current != self.value
        elif self.operator == "in":
            return current in self.value if isinstance(self.value, (list, tuple)) else False
        elif self.operator == "not_in":
            return current not in self.value if isinstance(self.value, (list, tuple)) else True
        elif self.operator == "contains":
            return self.value in current if isinstance(current, (str, list, tuple)) else False
        else:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }


class TriggerRule:
    """Trigger rule with conditions and response templates."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        priority: int,
        enabled: bool,
        conditions: list[TriggerCondition],
        templates: dict[str, str | None],
        cooldown_ms: int,
        interrupt_higher_priority: bool = False,
    ):
        """
        Initialize trigger rule.

        Args:
            rule_id: Unique rule identifier
            name: Human-readable name
            priority: Priority level (0=highest, 3=lowest)
            enabled: Whether the rule is enabled
            conditions: List of conditions that must all be true
            templates: Response templates for different movement states
            cooldown_ms: Cooldown period in milliseconds
            interrupt_higher_priority: Whether to interrupt higher priority triggers

        Raises:
            ValueError: If parameters are invalid (when using Pydantic validation)
        """
        # Validate using Pydantic model if available
        if PYDANTIC_AVAILABLE:
            # Convert TriggerCondition objects to dicts for validation
            conditions_data = [c.to_dict() for c in conditions]
            validated = TriggerRuleModel(
                rule_id=rule_id,
                name=name,
                priority=priority,
                enabled=enabled,
                conditions=conditions_data,
                templates=templates,
                cooldown_ms=cooldown_ms,
                interrupt_higher_priority=interrupt_higher_priority
            )
            self.id = validated.rule_id
            self.name = validated.name
            self.priority = validated.priority
            self.enabled = validated.enabled
            self.templates = validated.templates
            self.cooldown_ms = validated.cooldown_ms
            self.interrupt_higher_priority = validated.interrupt_higher_priority
            # Recreate conditions from validated data
            self.conditions = [
                TriggerCondition(**c.model_dump() if hasattr(c, 'model_dump') else c)
                for c in validated.conditions
            ]
        else:
            # Fallback to direct assignment
            self.id = rule_id
            self.name = name
            self.priority = priority
            self.enabled = enabled
            self.conditions = conditions
            self.templates = templates
            self.cooldown_ms = cooldown_ms
            self.interrupt_higher_priority = interrupt_higher_priority

        self.last_triggered_ms = 0

    def evaluate(self, state: dict[str, Any]) -> bool:
        """
        Evaluate all conditions against game state.

        Args:
            state: Game state dictionary

        Returns:
            True if all conditions are met
        """
        if not self.enabled:
            return False

        return all(condition.evaluate(state) for condition in self.conditions)

    def get_template(self, movement_state: str) -> str | None:
        """
        Get template for specific movement state.

        Args:
            movement_state: Movement state ("combat" or "non_combat")

        Returns:
            Template string or None if not available for this state
        """
        return self.templates.get(movement_state)

    def is_on_cooldown(self, current_time_ms: int) -> bool:
        """
        Check if trigger is on cooldown.

        Args:
            current_time_ms: Current time in milliseconds

        Returns:
            True if on cooldown
        """
        return (current_time_ms - self.last_triggered_ms) < self.cooldown_ms

    def update_last_triggered(self, current_time_ms: int) -> None:
        """
        Update last triggered timestamp.

        Args:
            current_time_ms: Current time in milliseconds
        """
        self.last_triggered_ms = current_time_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "conditions": [c.to_dict() for c in self.conditions],
            "templates": self.templates,
            "cooldown_ms": self.cooldown_ms,
            "interrupt_higher_priority": self.interrupt_higher_priority,
            "last_triggered_ms": self.last_triggered_ms,
        }
