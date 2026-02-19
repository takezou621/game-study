"""Trigger rule definition and condition evaluation."""

from typing import Dict, Any, Optional, List


class TriggerCondition:
    """Condition for trigger evaluation."""

    def __init__(self, field: str, operator: str, value: Any):
        """
        Initialize trigger condition.

        Args:
            field: Field path (e.g., "player.status.hp")
            operator: Comparison operator (eq, lt, gt, lte, gte, ne)
            value: Value to compare against
        """
        self.field = field
        self.operator = operator
        self.value = value

    def evaluate(self, state: Dict[str, Any]) -> bool:
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
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
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
        conditions: List[TriggerCondition],
        templates: Dict[str, Optional[str]],
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
        """
        self.id = rule_id
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.conditions = conditions
        self.templates = templates
        self.cooldown_ms = cooldown_ms
        self.interrupt_higher_priority = interrupt_higher_priority

        self.last_triggered_ms = 0

    def evaluate(self, state: Dict[str, Any]) -> bool:
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

    def get_template(self, movement_state: str) -> Optional[str]:
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

    def to_dict(self) -> Dict[str, Any]:
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
