"""Use case for evaluating triggers against game state."""

from dataclasses import dataclass
from typing import Protocol

from application.dto.trigger_dto import (
    TriggerEvaluationDTO,
    TriggerEvaluationResultDTO,
    TriggerResultDTO,
)
from domain.triggers.evaluator import DefaultTriggerEvaluator
from domain.triggers.policies.priority_policy import PriorityPolicy
from domain.triggers.rule import TriggerRule
from utils.time import get_timestamp_ms


class TriggerRepository(Protocol):
    """Protocol for trigger storage."""

    def get_all_rules(self) -> list[TriggerRule]:
        """Get all trigger rules."""
        ...

    def get_rule(self, rule_id: str) -> TriggerRule | None:
        """Get rule by ID."""
        ...


@dataclass
class EvaluateTriggersUseCase:
    """Use case for evaluating triggers against game state."""

    evaluator: DefaultTriggerEvaluator
    priority_policy: PriorityPolicy
    current_speech_priority: int | None = None

    def execute(self, input_dto: TriggerEvaluationDTO) -> TriggerEvaluationResultDTO:
        """Evaluate all triggers against game state.

        Args:
            input_dto: Input containing game state and context

        Returns:
            TriggerEvaluationResultDTO with firing triggers and selected trigger
        """
        # Add evaluation time to state for cooldown calculation
        state = input_dto.state.copy()
        state["_evaluation_time_ms"] = input_dto.evaluation_time_ms

        # Evaluate all triggers
        firing_triggers = self.evaluator.evaluate(state)

        if not firing_triggers:
            return TriggerEvaluationResultDTO(
                firing_triggers=[],
                selected_trigger=None,
                suppressed_count=0,
            )

        # Convert to DTOs
        trigger_results = [
            TriggerResultDTO(
                trigger_id=rule.id,
                trigger_name=rule.name,
                priority=rule.priority,
                template=rule.get_template(input_dto.movement_state),
                movement_state=input_dto.movement_state,
                should_interrupt=False,
                confidence=1.0,
                evaluation_time_ms=input_dto.evaluation_time_ms,
            )
            for rule in firing_triggers
        ]

        # Get highest priority trigger
        selected = self.evaluator.get_highest_priority(firing_triggers)

        if selected is None:
            return TriggerEvaluationResultDTO(
                firing_triggers=trigger_results,
                selected_trigger=None,
                suppressed_count=len(firing_triggers),
            )

        # Check if should interrupt current speech
        should_interrupt = False
        if input_dto.active_speech_priority is not None:
            should_interrupt = self.priority_policy.should_interrupt(
                new_priority=selected.priority,
                current_priority=input_dto.active_speech_priority,
            )

        # Find selected trigger in results
        selected_result = next(
            (t for t in trigger_results if t.trigger_id == selected.id),
            None,
        )

        if selected_result:
            selected_result.should_interrupt = should_interrupt

        # Sort by priority
        trigger_results.sort(key=lambda t: t.priority)

        return TriggerEvaluationResultDTO(
            firing_triggers=trigger_results,
            selected_trigger=selected_result,
            suppressed_count=len(firing_triggers) - 1,
        )

    def set_current_speech_priority(self, priority: int | None) -> None:
        """Set the priority of currently playing speech.

        Args:
            priority: Current speech priority or None if not playing
        """
        self.current_speech_priority = priority

    def mark_triggered(self, rule: TriggerRule) -> None:
        """Mark a rule as triggered (updates cooldown).

        Args:
            rule: Rule that was triggered
        """
        rule.update_last_triggered(get_timestamp_ms())

    @classmethod
    def create(cls, rules: list[TriggerRule]) -> "EvaluateTriggersUseCase":
        """Factory method to create use case with rules.

        Args:
            rules: List of trigger rules

        Returns:
            Configured EvaluateTriggersUseCase
        """
        evaluator = DefaultTriggerEvaluator(rules)
        priority_policy = PriorityPolicy()
        return cls(evaluator=evaluator, priority_policy=priority_policy)
