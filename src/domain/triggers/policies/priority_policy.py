"""Priority policy for trigger management."""

from dataclasses import dataclass
from enum import IntEnum


class Priority(IntEnum):
    """Trigger priority levels."""

    SURVIVAL = 0  # P0: Life-threatening situations
    TACTICAL = 1  # P1: Tactical advice
    LEARNING = 2  # P2: Learning opportunities
    CHATTER = 3  # P3: Casual conversation


@dataclass
class PriorityPolicy:
    """Policy for managing trigger priorities."""

    def should_interrupt(
        self,
        new_priority: int,
        current_priority: int,
        current_remaining_ms: int = 0,
    ) -> bool:
        """Determine if new trigger should interrupt current.

        Args:
            new_priority: Priority of new trigger
            current_priority: Priority of current speech
            current_remaining_ms: Remaining time for current speech

        Returns:
            True if should interrupt
        """
        # P0 always interrupts
        if new_priority == Priority.SURVIVAL:
            return True

        # P1 interrupts P2 and P3
        if new_priority == Priority.TACTICAL and current_priority >= Priority.LEARNING:
            return True

        # P2 interrupts P3 only if significant time remaining
        return bool(
            new_priority == Priority.LEARNING
            and current_priority == Priority.CHATTER
            and current_remaining_ms > 2000
        )

    def get_priority_label(self, priority: int) -> str:
        """Get human-readable label for priority.

        Args:
            priority: Priority level

        Returns:
            Label string
        """
        labels = {
            Priority.SURVIVAL: "Survival",
            Priority.TACTICAL: "Tactical",
            Priority.LEARNING: "Learning",
            Priority.CHATTER: "Chatter",
        }
        return labels.get(priority, f"P{priority}")

    def get_max_response_duration_ms(self, priority: int) -> int:
        """Get maximum response duration for priority level.

        Lower priority = shorter responses.

        Args:
            priority: Priority level

        Returns:
            Maximum response duration in milliseconds
        """
        durations = {
            Priority.SURVIVAL: 3000,  # Short, urgent
            Priority.TACTICAL: 5000,  # Medium
            Priority.LEARNING: 10000,  # Longer explanations
            Priority.CHATTER: 8000,  # Medium-long
        }
        return durations.get(priority, 5000)

    def get_template_preference(self, priority: int) -> str:
        """Get preferred template style for priority level.

        Args:
            priority: Priority level

        Returns:
            Template style preference
        """
        if priority <= Priority.TACTICAL:
            return "short"  # Short, concise templates
        return "normal"  # Normal length templates

    def compare(self, priority_a: int, priority_b: int) -> int:
        """Compare two priorities.

        Args:
            priority_a: First priority
            priority_b: Second priority

        Returns:
            Negative if a is higher priority, positive if b is higher, 0 if equal
        """
        return priority_a - priority_b
