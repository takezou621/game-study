"""Event handler for Realtime API responses."""

import base64
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Realtime API event types."""

    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CONVERSATION_CREATED = "conversation.created"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
        "conversation.item.input_audio_transcription.completed"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = (
        "conversation.item.input_audio_transcription.failed"
    )
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    ERROR = "error"


@dataclass
class ResponseAccumulator:
    """Accumulates response data from multiple events."""

    text: str = ""
    transcript: str = ""
    audio_chunks: list[bytes] = field(default_factory=list)
    start_time: float | None = None
    is_complete: bool = False

    def add_text_delta(self, delta: str) -> None:
        """Add text delta."""
        self.text += delta

    def add_transcript_delta(self, delta: str) -> None:
        """Add transcript delta."""
        self.transcript += delta

    def add_audio_delta(self, audio_base64: str) -> None:
        """Add audio delta."""
        self.audio_chunks.append(base64.b64decode(audio_base64))

    def get_audio(self) -> bytes | None:
        """Get combined audio data."""
        if not self.audio_chunks:
            return None
        return b"".join(self.audio_chunks)

    def get_duration_ms(self) -> int:
        """Estimate audio duration."""
        audio = self.get_audio()
        if audio is None:
            return 0
        # PCM16 at 24kHz = 48000 bytes/second
        return len(audio) // 48 * 1000


class RealtimeEventHandler:
    """
    Handles events from Realtime API.

    Processes incoming events and accumulates response data.
    """

    def __init__(
        self,
        on_text_callback: Callable[[str], None] | None = None,
        on_audio_callback: Callable[[bytes], None] | None = None,
        on_complete_callback: Callable[[ResponseAccumulator], None] | None = None,
        on_error_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialize event handler.

        Args:
            on_text_callback: Called for text deltas
            on_audio_callback: Called for audio chunks
            on_complete_callback: Called when response complete
            on_error_callback: Called on errors
        """
        self.on_text_callback = on_text_callback
        self.on_audio_callback = on_audio_callback
        self.on_complete_callback = on_complete_callback
        self.on_error_callback = on_error_callback

        self._accumulator = ResponseAccumulator()

    def handle_event(self, event: dict[str, Any]) -> ResponseAccumulator | None:
        """
        Handle an event from the API.

        Args:
            event: Event dictionary

        Returns:
            ResponseAccumulator if response complete, None otherwise
        """
        event_type = event.get("type")

        if event_type == EventType.RESPONSE_TEXT_DELTA.value:
            delta = event.get("delta", "")
            self._accumulator.add_text_delta(delta)
            if self.on_text_callback:
                self.on_text_callback(delta)

        elif event_type == EventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA.value:
            delta = event.get("delta", "")
            self._accumulator.add_transcript_delta(delta)

        elif event_type == EventType.RESPONSE_AUDIO_DELTA.value:
            audio_base64 = event.get("delta", "")
            self._accumulator.add_audio_delta(audio_base64)
            if self.on_audio_callback:
                audio_data = base64.b64decode(audio_base64)
                self.on_audio_callback(audio_data)

        elif event_type in (
            EventType.RESPONSE_AUDIO_DONE.value,
            EventType.RESPONSE_DONE.value,
        ):
            self._accumulator.is_complete = True
            if self.on_complete_callback:
                self.on_complete_callback(self._accumulator)
            return self._accumulator

        elif event_type == EventType.ERROR.value:
            error = event.get("error", {})
            message = error.get("message", "Unknown error")
            logger.error(f"Realtime API error: {message}")
            if self.on_error_callback:
                self.on_error_callback(message)

        return None

    def reset(self) -> None:
        """Reset accumulator for new response."""
        self._accumulator = ResponseAccumulator()

    def get_accumulator(self) -> ResponseAccumulator:
        """Get current accumulator."""
        return self._accumulator
