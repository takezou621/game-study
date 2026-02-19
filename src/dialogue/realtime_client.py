"""OpenAI Realtime API client for voice conversation."""

import os
import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class VoiceResponse:
    """Voice response from Realtime API."""
    text: str
    audio_data: Optional[bytes] = None
    duration_ms: Optional[int] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RealtimeVoiceClient:
    """
    OpenAI Realtime API client for voice conversations.

    This client manages WebSocket connections to OpenAI's Realtime API,
    handles audio I/O, and provides low-latency voice responses.

    MVP Implementation:
    - Text → Speech (TTS) with queue-based playback
    - Cooldown and interrupt control
    - Priority-based response management

    Future Enhancements:
    - Speech → Text (STT) for player voice input
    - Bidirectional conversation
    - WebRTC streaming integration
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        system_prompt_path: Optional[str] = None,
        cooldown_ms: int = 3000,  # Minimum time between responses
        max_response_length_ms: int = 10000,  # Maximum response duration
        enable_audio_output: bool = True
    ):
        """
        Initialize Realtime Voice client.

        Args:
            api_key: OpenAI API key
            model: Realtime model to use
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            system_prompt_path: Path to system prompt file
            cooldown_ms: Minimum cooldown between responses
            max_response_length_ms: Maximum response duration
            enable_audio_output: Enable audio output (True) or text-only (False)
        """
        if not OPENAI_AVAILABLE:
            self.client = None
            self.enabled = False
            self.api_key = None
            self.model = model
            self.voice = voice
            self.system_prompt = self._load_system_prompt(system_prompt_path)
            self.cooldown_ms = cooldown_ms
            self.max_response_length_ms = max_response_length_ms
            self.enable_audio_output = enable_audio_output
            self.session = None
            self.is_speaking = False
            self.last_spoken_time = 0.0
            self.response_queue = None
            self.interrupt_requested = False
            self.loop = None
            return

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.enabled = False
            self.client = None
            self.model = model
            self.voice = voice
            self.system_prompt = self._load_system_prompt(system_prompt_path)
            self.cooldown_ms = cooldown_ms
            self.max_response_length_ms = max_response_length_ms
            self.enable_audio_output = enable_audio_output
            self.session = None
            self.is_speaking = False
            self.last_spoken_time = 0.0
            self.response_queue = None
            self.interrupt_requested = False
            self.loop = None
            return

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.voice = voice
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.cooldown_ms = cooldown_ms
        self.max_response_length_ms = max_response_length_ms
        self.enable_audio_output = enable_audio_output

        # State management
        self.enabled = True
        self.session: Optional[Any] = None
        self.is_speaking = False
        self.last_spoken_time = 0.0
        self.response_queue: asyncio.Queue[VoiceResponse] = asyncio.Queue()
        self.interrupt_requested = False

        # Event loop for async operations
        self.loop = asyncio.new_event_loop()

    def _load_system_prompt(self, path: Optional[str]) -> str:
        """Load system prompt from file."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()

        # Default system prompt for voice coaching
        return """You are an AI English teacher and gaming coach for Fortnite players. Your goal is to help players improve their English skills while playing the game naturally.

## Voice Communication Guidelines

- **Keep responses SHORT**: 1-2 sentences during combat, 2-3 sentences outside combat
- **Speak naturally**: Use clear, gaming-appropriate English
- **Don't over-explain**: Players are focused on the game
- **Use short phrases**: "Low HP! Heal!" instead of "Your health is very low, you should heal"

## Priority Levels

- **P0 (Survival)**: Urgent commands. Be direct and loud.
- **P1 (Tactical)**: Strategic suggestions. Be informative but brief.
- **P2 (Learning)**: Vocabulary lessons. Only when safe.
- **P3 (Chatter)**: Casual conversation. Very short.

## Combat vs Non-Combat

- **Combat**: Maximum 2 sentences. Focus on survival.
- **Non-Combat**: 2-4 sentences. You can explain briefly.

## Example Responses

P0 - Low HP in combat: "Low HP! Get cover now!"
P1 - Storm shrinking: "Storm is moving. Rotate to the safe zone."
P2 - Weapon pickup: "That's a Legendary assault rifle. Great for medium range."
P3 - Small talk: "How's it going?"
"""

    async def initialize_session(self) -> bool:
        """
        Initialize Realtime API session.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Create a realtime session
            # Note: The exact API will depend on OpenAI's Realtime API implementation
            # This is a placeholder structure based on expected API

            self.session = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}],
                audio={"voice": self.voice, "format": "pcm16"},
                temperature=0.7
            )

            return True

        except Exception as e:
            print(f"Failed to initialize Realtime session: {e}")
            return False

    def speak(
        self,
        text: str,
        priority: int = 2,
        allow_interrupt: bool = True
    ) -> Optional[VoiceResponse]:
        """
        Speak text with voice synthesis.

        Synchronous wrapper for async speech generation.

        Args:
            text: Text to speak
            priority: Priority level (0=highest, 3=lowest)
            allow_interrupt: Allow this speech to interrupt current speech

        Returns:
            VoiceResponse object or None if failed
        """
        if not self.enabled:
            # Return text-only response if audio disabled
            return VoiceResponse(text=text)

        # Check cooldown
        time_since_last = (time.time() - self.last_spoken_time) * 1000
        if time_since_last < self.cooldown_ms and priority > 1:
            # Skip low-priority speech during cooldown
            return None

        # Interrupt current speech if requested and allowed
        if allow_interrupt and self.is_speaking:
            self.interrupt_requested = True

        # Run async speech generation in event loop
        try:
            response = self.loop.run_until_complete(
                self._generate_speech(text)
            )
            self.last_spoken_time = time.time()
            return response
        except Exception as e:
            print(f"Speech generation failed: {e}")
            return VoiceResponse(text=text)

    async def _generate_speech(self, text: str) -> VoiceResponse:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech

        Returns:
            VoiceResponse with audio data
        """
        if not self.enable_audio_output:
            return VoiceResponse(text=text)

        try:
            # Use OpenAI's TTS API as MVP
            # In Phase 2+, we'll use Realtime API directly
            from openai import AsyncOpenAI
            tts_client = AsyncOpenAI(api_key=self.api_key)

            response = await tts_client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text[:500]  # Limit text length
            )

            audio_data = await response.aread()

            return VoiceResponse(
                text=text,
                audio_data=audio_data,
                duration_ms=len(audio_data) // 32  # Approximate for 16kHz
            )

        except Exception as e:
            print(f"TTS generation failed: {e}")
            return VoiceResponse(text=text)

    def speak_with_trigger(
        self,
        trigger_info: Dict[str, Any],
        state: Dict[str, Any],
        movement_state: str
    ) -> Optional[VoiceResponse]:
        """
        Generate and speak response based on trigger.

        Args:
            trigger_info: Trigger information (id, name, priority, template)
            state: Current game state
            movement_state: Movement state ("combat" or "non_combat")

        Returns:
            VoiceResponse or None
        """
        if not self.enabled:
            return None

        priority = trigger_info.get('priority', 2)

        # Get or generate text
        template = trigger_info.get('template')
        if template:
            text = self._enhance_template(template, state, movement_state)
        else:
            text = self._generate_text(trigger_info, state, movement_state)

        if not text:
            return None

        # P0 triggers should always be allowed to interrupt
        allow_interrupt = (priority == 0)

        return self.speak(text, priority=priority, allow_interrupt=allow_interrupt)

    def _enhance_template(
        self,
        template: str,
        state: Dict[str, Any],
        movement_state: str
    ) -> str:
        """Enhance template with state information."""
        # MVP: Return template as-is
        # In Phase 2+, add context from state
        return template

    def _generate_text(
        self,
        trigger_info: Dict[str, Any],
        state: Dict[str, Any],
        movement_state: str
    ) -> str:
        """Generate text response (fallback if no template)."""
        return f"Response: {trigger_info.get('name', 'Unknown')}"

    def stop(self) -> None:
        """Stop current speech."""
        self.interrupt_requested = True
        self.is_speaking = False

    def shutdown(self) -> None:
        """Shutdown client and cleanup resources."""
        self.stop()
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        self.session = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


# Convenience function for creating a client
def create_voice_client(
    api_key: Optional[str] = None,
    voice: str = "alloy",
    enable_audio: bool = True
) -> RealtimeVoiceClient:
    """
    Create a Realtime Voice client with sensible defaults.

    Args:
        api_key: OpenAI API key
        voice: Voice to use
        enable_audio: Enable audio output

    Returns:
        Configured RealtimeVoiceClient instance
    """
    return RealtimeVoiceClient(
        api_key=api_key,
        voice=voice,
        enable_audio_output=enable_audio,
        cooldown_ms=3000,
        max_response_length_ms=10000
    )
