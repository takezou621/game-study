"""Trigger evaluator for evaluating trigger rules against game state."""

from typing import Any, Protocol

from domain.triggers.rule import TriggerRule


class TriggerEvaluator(Protocol):
    """Protocol for trigger evaluation."""

    def evaluate(self, state: dict[str, Any]) -> list[TriggerRule]:
        """Evaluate all triggers against state and return firing triggers."""
        ...

    def should_interrupt(self, new_priority: int, current_priority: int) -> bool:
        """Determine if new trigger should interrupt current."""
        ...


class DefaultTriggerEvaluator:
    """Default implementation of trigger evaluator."""

    def __init__(self, rules: list[TriggerRule]):
        """Initialize with list of rules.

        Args:
            rules: List of trigger rules to evaluate
        """
        self.rules = sorted(rules, key=lambda r: r.priority)

    def evaluate(self, state: dict[str, Any]) -> list[TriggerRule]:
        """Evaluate all triggers and return list of firing triggers.

        Args:
            state: Current game state

        Returns:
            List of triggers that fired (conditions met, not on cooldown)
        """
        firing = []
        current_time_ms = state.get("_evaluation_time_ms", 0)

        for rule in self.rules:
            if rule.evaluate(state) and not rule.is_on_cooldown(current_time_ms):
                firing.append(rule)

        return firing

    def should_interrupt(self, new_priority: int, current_priority: int) -> bool:
        """Determine if new trigger should interrupt current speech.

        Args:
            new_priority: Priority of new trigger (0=highest)
            current_priority: Priority of currently playing trigger

        Returns:
            True if new trigger should interrupt
        """
        # P0 (survival) always interrupts
        if new_priority == 0:
            return True

        # P1 interrupts P2 and P3
        return bool(new_priority == 1 and current_priority >= 2)

    def get_highest_priority(self, triggers: list[TriggerRule]) -> TriggerRule | None:
        """Get highest priority trigger from list.

        Args:
            triggers: List of firing triggers

        Returns:
            Highest priority trigger or None if empty
        """
        if not triggers:
            return None

        return min(triggers, key=lambda t: t.priority)

    def add_rule(self, rule: TriggerRule) -> None:
        """Add a rule to the evaluator.

        Args:
            rule: Rule to add
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                self.rules.pop(i)
                return True
        return False

    def get_rule(self, rule_id: str) -> TriggerRule | None:
        """Get rule by ID.

        Args:
            rule_id: ID of rule to get

        Returns:
            Rule or None if not found
        """
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
