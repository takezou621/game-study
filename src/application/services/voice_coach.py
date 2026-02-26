"""Voice coach service for coordinating voice responses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from application.dto.response_dto import AudioResponseDTO, QueuedResponseDTO
from application.usecases.evaluate_triggers import EvaluateTriggersUseCase
from application.usecases.generate_response import GenerateResponseUseCase
from domain.entities.game_state import GameState
from domain.triggers.policies.priority_policy import PriorityPolicy
from utils.time import get_timestamp_ms


class CoachState(str, Enum):
    """Voice coach state."""

    IDLE = "idle"
    SPEAKING = "speaking"
    PROCESSING = "processing"


@dataclass
class VoiceCoachService:
    """Service for coordinating voice coaching responses."""

    evaluate_usecase: EvaluateTriggersUseCase
    generate_usecase: GenerateResponseUseCase
    priority_policy: PriorityPolicy = field(default_factory=PriorityPolicy)

    # State
    state: CoachState = CoachState.IDLE
    current_response: AudioResponseDTO | None = None
    response_queue: list[QueuedResponseDTO] = field(default_factory=list)
    last_response_ms: int = 0

    # Configuration
    max_queue_size: int = 3
    min_response_interval_ms: int = 2000

    def process_state(
        self,
        game_state: GameState,
        force: bool = False,
    ) -> AudioResponseDTO | None:
        """Process game state and potentially generate response.

        Args:
            game_state: Current game state
            force: Force evaluation even if recently responded

        Returns:
            AudioResponseDTO if response generated, None otherwise
        """
        current_time = get_timestamp_ms()

        # Check if enough time has passed
        if not force and not self._should_respond(game_state, current_time):
            return None

        # Evaluate triggers
        from application.dto.trigger_dto import TriggerEvaluationDTO

        evaluation_input = TriggerEvaluationDTO(
            state=game_state.to_dict(),
            movement_state=game_state.movement_state,
            evaluation_time_ms=current_time,
            active_speech_priority=self._get_current_priority(),
        )

        evaluation_result = self.evaluate_usecase.execute(evaluation_input)

        if not evaluation_result.has_firing_triggers:
            return None

        selected = evaluation_result.selected_trigger
        if selected is None:
            return None

        # Check if should interrupt current speech
        if self.state == CoachState.SPEAKING and selected.should_interrupt:
            self._interrupt_current()

        # Generate response
        response = self.generate_usecase.execute_from_trigger(
            trigger=selected,
            state=game_state.to_dict(),
            use_llm=False,  # Use templates for now
        )

        if response is None:
            return None

        # Convert to audio
        audio_response = self.generate_usecase.execute_text_to_speech(response)

        if audio_response is None:
            # Return text-only response if TTS unavailable
            return AudioResponseDTO(
                text=response.text,
                priority=response.priority,
                source_trigger=response.source_trigger,
            )

        # Mark trigger as triggered
        trigger_rule = self.evaluate_usecase.evaluator.get_rule(selected.trigger_id)
        if trigger_rule:
            self.evaluate_usecase.mark_triggered(trigger_rule)

        # Update state
        self.last_response_ms = current_time

        # Queue or play directly
        if self.state == CoachState.SPEAKING:
            self._queue_response(audio_response)
            return None
        else:
            self._start_playback(audio_response)
            return audio_response

    def _should_respond(self, state: GameState, current_time: int) -> bool:
        """Check if should generate a response.

        Args:
            state: Current game state
            current_time: Current timestamp

        Returns:
            True if should respond
        """
        elapsed = current_time - self.last_response_ms

        # Always respond to urgent situations
        if state.needs_attention and elapsed >= self.min_response_interval_ms:
            return True

        # Otherwise respect minimum interval
        return elapsed >= self.min_response_interval_ms

    def _get_current_priority(self) -> int | None:
        """Get priority of currently playing response."""
        if self.state != CoachState.SPEAKING or self.current_response is None:
            return None
        return self.current_response.priority

    def _interrupt_current(self) -> None:
        """Interrupt current speech."""
        if self.current_response is not None:
            self.current_response.interrupted = True
            self.state = CoachState.IDLE
            self.current_response = None

    def _queue_response(self, response: AudioResponseDTO) -> bool:
        """Add response to queue.

        Args:
            response: Response to queue

        Returns:
            True if queued, False if queue full
        """
        if len(self.response_queue) >= self.max_queue_size:
            # Remove lowest priority item if new is higher priority
            lowest = max(self.response_queue, key=lambda q: q.response.priority)
            if response.priority < lowest.response.priority:
                self.response_queue.remove(lowest)
            else:
                return False

        queued = QueuedResponseDTO(
            response=response,
            queue_position=len(self.response_queue),
        )
        self.response_queue.append(queued)
        return True

    def _start_playback(self, response: AudioResponseDTO) -> None:
        """Start playback of response.

        Args:
            response: Response to play
        """
        response.playback_started_ms = get_timestamp_ms()
        self.current_response = response
        self.state = CoachState.SPEAKING

    def mark_playback_complete(self) -> AudioResponseDTO | None:
        """Mark current playback as complete and return next queued.

        Returns:
            Next AudioResponseDTO to play, or None if queue empty
        """
        if self.current_response is not None:
            self.current_response.playback_completed_ms = get_timestamp_ms()

        self.current_response = None
        self.state = CoachState.IDLE

        # Get next from queue
        if self.response_queue:
            next_response = self.response_queue.pop(0).response
            self._start_playback(next_response)
            return next_response

        return None

    def get_status(self) -> dict[str, Any]:
        """Get current coach status.

        Returns:
            Status dictionary
        """
        return {
            "state": self.state.value,
            "current_response": self.current_response.to_dict() if self.current_response else None,
            "queue_size": len(self.response_queue),
            "last_response_ms": self.last_response_ms,
        }

    def clear_queue(self) -> int:
        """Clear the response queue.

        Returns:
            Number of items cleared
        """
        count = len(self.response_queue)
        self.response_queue.clear()
        return count

    def shutdown(self) -> None:
        """Shutdown the coach service."""
        self._interrupt_current()
        self.clear_queue()
        self.state = CoachState.IDLE
