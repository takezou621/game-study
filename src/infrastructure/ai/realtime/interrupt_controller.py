"""Interrupt controller for voice responses."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class InterruptPriority(int, Enum):
    """Priority levels for interrupt decisions."""

    SURVIVAL = 0  # P0: Life-threatening
    TACTICAL = 1  # P1: Tactical advice
    LEARNING = 2  # P2: Learning opportunities
    CHATTER = 3  # P3: Casual conversation


@dataclass
class InterruptDecision:
    """Result of interrupt evaluation."""

    should_interrupt: bool
    new_priority: int
    current_priority: int
    reason: str = ""


class InterruptController:
    """
    Controls speech interruption based on priority.

    Determines when new speech should interrupt current speech
    based on priority levels and timing.
    """

    def __init__(
        self,
        on_interrupt_callback: Callable[[], None] | None = None,
    ):
        """
        Initialize interrupt controller.

        Args:
            on_interrupt_callback: Called when interrupt occurs
        """
        self.on_interrupt_callback = on_interrupt_callback
        self._current_priority: int = 99  # No current speech
        self._interrupt_requested = False

    @property
    def current_priority(self) -> int:
        """Get current speech priority."""
        return self._current_priority

    @property
    def is_interrupt_requested(self) -> bool:
        """Check if interrupt has been requested."""
        return self._interrupt_requested

    def set_current_priority(self, priority: int) -> None:
        """
        Set current speech priority.

        Args:
            priority: Priority of current speech
        """
        self._current_priority = priority

    def clear_current_priority(self) -> None:
        """Clear current speech priority."""
        self._current_priority = 99

    def should_interrupt(self, new_priority: int) -> InterruptDecision:
        """
        Determine if new speech should interrupt current.

        Args:
            new_priority: Priority of new speech

        Returns:
            InterruptDecision with result
        """
        current = self._current_priority

        # No current speech - no need to interrupt
        if current >= 99:
            return InterruptDecision(
                should_interrupt=False,
                new_priority=new_priority,
                current_priority=current,
                reason="No current speech",
            )

        # P0 always interrupts
        if new_priority == InterruptPriority.SURVIVAL:
            return InterruptDecision(
                should_interrupt=True,
                new_priority=new_priority,
                current_priority=current,
                reason="P0 (survival) always interrupts",
            )

        # P1 interrupts P2 and P3
        if new_priority == InterruptPriority.TACTICAL and current >= InterruptPriority.LEARNING:
            return InterruptDecision(
                should_interrupt=True,
                new_priority=new_priority,
                current_priority=current,
                reason="P1 interrupts P2/P3",
            )

        # Lower priority cannot interrupt higher
        if new_priority >= current:
            return InterruptDecision(
                should_interrupt=False,
                new_priority=new_priority,
                current_priority=current,
                reason="Lower or equal priority cannot interrupt",
            )

        return InterruptDecision(
            should_interrupt=False,
            new_priority=new_priority,
            current_priority=current,
            reason="Priority rules do not allow interrupt",
        )

    def request_interrupt(self, priority: int) -> bool:
        """
        Request an interrupt if allowed.

        Args:
            priority: Priority of new speech

        Returns:
            True if interrupt was requested
        """
        decision = self.should_interrupt(priority)

        if decision.should_interrupt:
            self._interrupt_requested = True
            if self.on_interrupt_callback:
                self.on_interrupt_callback()
            return True

        return False

    def clear_interrupt(self) -> None:
        """Clear interrupt request."""
        self._interrupt_requested = False

    def reset(self) -> None:
        """Reset controller state."""
        self._current_priority = 99
        self._interrupt_requested = False
