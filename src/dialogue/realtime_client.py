"""OpenAI Realtime API client for voice conversation."""

import asyncio
import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import constants with fallback for standalone usage
try:
    from constants import (
        DEFAULT_COOLDOWN_MS,
        DEFAULT_MAX_RESPONSE_LENGTH_MS,
        MAX_TEXT_LENGTH,
    )
except ImportError:
    DEFAULT_COOLDOWN_MS = 3000
    DEFAULT_MAX_RESPONSE_LENGTH_MS = 10000
    MAX_TEXT_LENGTH = 500


class SpeechState(Enum):
    """Current speech state."""
    IDLE = "idle"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class VoiceResponse:
    """Voice response from Realtime API."""
    text: str
    audio_data: bytes | None = None
    duration_ms: int | None = None
    timestamp: float = None
    priority: int = 2
    interrupted: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class AudioChunk:
    """Audio chunk for playback queue."""
    data: bytes
    timestamp: float = field(default_factory=time.time)


class RealtimeVoiceClient:
    """
    OpenAI Realtime API client for voice conversations.

    This client manages WebSocket connections to OpenAI's Realtime API,
    handles audio I/O, and provides low-latency voice responses.

    Features:
    - WebSocket-based Realtime API connection
    - Interrupt control (P0 triggers always have priority)
    - Short response templates during combat
    - Audio playback queue
    - Cooldown management
    """

    # Short response templates for combat situations
    COMBAT_TEMPLATES = {
        0: {  # P0 - Survival
            "low_hp": "Low HP! Cover!",
            "knocked": "Knocked! Ping!",
            "storm": "Storm! Move!",
        },
        1: {  # P1 - Tactical
            "rotate": "Rotate now!",
            "storm_shrinking": "Storm moving!",
        }
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        voice: str = "alloy",
        system_prompt_path: str | None = None,
        cooldown_ms: int = DEFAULT_COOLDOWN_MS,
        max_response_length_ms: int = DEFAULT_MAX_RESPONSE_LENGTH_MS,
        enable_audio_output: bool = True,
        use_realtime_api: bool = True
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
            use_realtime_api: Use Realtime API (True) or TTS API (False)

        Raises:
            ValueError: If API key is not available when required
        """
        self.model = model
        self.voice = voice
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.cooldown_ms = cooldown_ms
        self.max_response_length_ms = max_response_length_ms
        self.enable_audio_output = enable_audio_output
        self.use_realtime_api = use_realtime_api and WEBSOCKETS_AVAILABLE

        # State management
        self.speech_state = SpeechState.IDLE
        self.last_spoken_time = 0.0
        self.current_priority = 99  # Lower is higher priority
        self.interrupt_requested = False

        # Client initialization
        self.client: AsyncOpenAI | None = None
        self._api_key_validated = False  # Track if API key was validated

        if not OPENAI_AVAILABLE:
            self.enabled = False
            return

        # Get API key and pass directly to client (do not store long-term)
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            self.enabled = False
            logger.warning("OpenAI API key not found - voice features disabled")
            return

        # Validate API key format
        if resolved_api_key and not resolved_api_key.startswith('sk-'):
            logger.warning("API key format appears invalid (should start with 'sk-')")

        self._api_key_validated = True
        self._resolved_api_key = resolved_api_key  # Store for realtime API connection
        self.client = AsyncOpenAI(api_key=resolved_api_key)
        self.enabled = True

        # WebSocket connection
        self.ws: Any | None = None
        self.ws_connected = False
        self.ws_lock = threading.Lock()

        # Audio playback queue
        self.audio_queue: Queue[AudioChunk] = Queue()
        self.playback_thread: threading.Thread | None = None
        self.playback_running = False

        # Event loop for async operations
        self.loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

        # Response queue for async responses
        self.response_queue: Queue[VoiceResponse] = Queue()

        # Initialize event loop in separate thread
        if self.enabled:
            self._start_event_loop()

    def _get_api_key(self) -> str:
        """
        Get API key from stored value or environment.

        Returns:
            API key string

        Raises:
            ValueError: If API key is not available
        """
        # First check stored key from constructor
        api_key = getattr(self, '_resolved_api_key', None)
        if api_key:
            return api_key

        # Fall back to environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment or constructor")
        return api_key

    def _load_system_prompt(self, path: str | None) -> str:
        """Load system prompt from file."""
        if path and os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()

        return """You are an AI English teacher and gaming coach for Fortnite players.

## Voice Communication Guidelines

- **Keep responses SHORT**: 1-2 sentences during combat, 2-3 sentences outside combat
- **Speak naturally**: Use clear, gaming-appropriate English
- **Don't over-explain**: Players are focused on the game

## Priority Levels

- **P0 (Survival)**: Urgent commands. Maximum 5 words.
- **P1 (Tactical)**: Strategic suggestions. Maximum 10 words.
- **P2 (Learning)**: Vocabulary lessons. Only when safe. 2-3 sentences.
- **P3 (Chatter)**: Casual conversation. Very short.

## Combat Response Rules

During combat, ALWAYS use the shortest possible response:
- "Low HP! Cover!" instead of "Your health is low, find cover"
- "Storm! Move!" instead of "The storm is coming, you should move"
"""

    def _start_event_loop(self):
        """Start event loop in separate thread."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)

    def _get_short_response(
        self,
        trigger_info: dict[str, Any],
        movement_state: str
    ) -> str | None:
        """
        Get short response for combat situations.

        Args:
            trigger_info: Trigger information
            movement_state: Current movement state

        Returns:
            Short template or None
        """
        if movement_state != "combat":
            return None

        priority = trigger_info.get('priority', 2)
        trigger_id = trigger_info.get('rule_id', '')

        if priority in self.COMBAT_TEMPLATES:
            for key, template in self.COMBAT_TEMPLATES[priority].items():
                if key in trigger_id:
                    return template

        return None

    async def _connect_realtime_api(self) -> bool:
        """
        Connect to OpenAI Realtime API via WebSocket.

        Returns:
            True if connected successfully
        """
        if not WEBSOCKETS_AVAILABLE:
            return False

        try:
            url = "wss://api.openai.com/v1/realtime"
            # Get API key dynamically from environment for WebSocket connection
            api_key = self._get_api_key()
            headers = [
                ("Authorization", f"Bearer {api_key}"),
                ("OpenAI-Beta", "realtime=v1")
            ]

            self.ws = await websockets.connect(
                url,
                extra_headers=headers
            )

            # Configure session
            session_config = {
                "type": "session.update",
                "session": {
                    "model": self.model,
                    "voice": self.voice,
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "instructions": self.system_prompt,
                    "max_response_output_tokens": 150,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    }
                }
            }

            await self.ws.send(json.dumps(session_config))

            # Wait for session.created event
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            event = json.loads(response)

            if event.get("type") == "session.created":
                self.ws_connected = True
                return True

        except Exception:
            logger.error("Failed to initialize Realtime session", exc_info=True)
            return False

    async def _disconnect_realtime_api(self):
        """Disconnect from Realtime API."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
            self.ws_connected = False

    async def _send_text_and_receive_audio(
        self,
        text: str,
        priority: int
    ) -> VoiceResponse:
        """
        Send text and receive audio response via Realtime API.

        Args:
            text: Text to speak
            priority: Priority level

        Returns:
            VoiceResponse with audio data
        """
        if not self.ws_connected:
            # Fallback to TTS API
            return await self._generate_speech_tts(text)

        try:
            # Send text message
            message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }

            await self.ws.send(json.dumps(message))

            # Request response
            response_request = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Respond briefly and naturally."
                }
            }

            await self.ws.send(json.dumps(response_request))

            # Collect response
            audio_chunks = []
            response_text = ""
            start_time = time.time()

            while True:
                try:
                    raw = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    event = json.loads(raw)
                    event_type = event.get("type")

                    if event_type == "response.audio_transcript.delta":
                        response_text += event.get("delta", "")

                    elif event_type == "response.audio.delta":
                        audio_base64 = event.get("delta", "")
                        audio_chunks.append(base64.b64decode(audio_base64))

                    elif event_type == "response.audio.done" or event_type == "response.done":
                        break

                    # Timeout check
                    if (time.time() - start_time) * 1000 > self.max_response_length_ms:
                        await self._cancel_response()
                        break

                except asyncio.TimeoutError:
                    break

            # Combine audio chunks
            audio_data = b"".join(audio_chunks) if audio_chunks else None
            duration_ms = len(audio_data) // 32 if audio_data else 0

            return VoiceResponse(
                text=response_text or text,
                audio_data=audio_data,
                duration_ms=duration_ms,
                priority=priority
            )

        except Exception as e:
            logger.error("Realtime API error: %s", e)
            return await self._generate_speech_tts(text)

    async def _cancel_response(self):
        """Cancel current response generation."""
        if self.ws and self.ws_connected:
            try:
                await self.ws.send(json.dumps({"type": "response.cancel"}))
            except Exception:
                pass

    async def _generate_speech_tts(self, text: str) -> VoiceResponse:
        """
        Generate speech using TTS API (fallback).

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
            # Use the existing client instead of creating a new one
            response = await self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text[:MAX_TEXT_LENGTH]
            )

            audio_data = await response.aread()

            return VoiceResponse(
                text=text,
                audio_data=audio_data,
                duration_ms=len(audio_data) // 32
            )

        except Exception:
            logger.error("TTS generation failed", exc_info=True)
            return VoiceResponse(text=text)

    def speak(
        self,
        text: str,
        priority: int = 2,
        allow_interrupt: bool = True
    ) -> VoiceResponse | None:
        """
        Speak text with voice synthesis.

        Args:
            text: Text to speak
            priority: Priority level (0=highest, 3=lowest)
            allow_interrupt: Allow this speech to interrupt current speech

        Returns:
            VoiceResponse object or None if failed
        """
        if not self.enabled:
            return VoiceResponse(text=text)

        # Check cooldown for low priority
        time_since_last = (time.time() - self.last_spoken_time) * 1000
        if time_since_last < self.cooldown_ms and priority > 1:
            return None

        # Interrupt handling
        if allow_interrupt and priority < self.current_priority:
            self.interrupt_requested = True
            self.speech_state = SpeechState.INTERRUPTED

        # Update state
        self.current_priority = priority
        self.speech_state = SpeechState.SPEAKING

        try:
            if self.loop is None:
                return VoiceResponse(text=text)

            # Run async speech generation
            future = asyncio.run_coroutine_threadsafe(
                self._generate_speech(text, priority),
                self.loop
            )
            response = future.result(timeout=30.0)

            self.last_spoken_time = time.time()
            self.speech_state = SpeechState.IDLE
            self.current_priority = 99

            return response

        except Exception as e:
            logger.error("Speech generation failed: %s", e)
            self.speech_state = SpeechState.IDLE
            return VoiceResponse(text=text)

    async def _generate_speech(self, text: str, priority: int) -> VoiceResponse:
        """Generate speech using appropriate API."""
        if self.use_realtime_api:
            if not self.ws_connected:
                await self._connect_realtime_api()

            if self.ws_connected:
                return await self._send_text_and_receive_audio(text, priority)

        return await self._generate_speech_tts(text)

    def speak_with_trigger(
        self,
        trigger_info: dict[str, Any],
        state: dict[str, Any],
        movement_state: str
    ) -> VoiceResponse | None:
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

        # Check for short response during combat
        short_response = self._get_short_response(trigger_info, movement_state)
        if short_response:
            return self.speak(short_response, priority=priority, allow_interrupt=True)

        # Get template based on movement state
        template = trigger_info.get('template')
        if isinstance(template, dict):
            text = template.get(movement_state) or template.get('non_combat')
        else:
            text = template

        if not text:
            return None

        # Enhance template with state
        text = self._enhance_template(text, state, movement_state)

        # P0 triggers should always be allowed to interrupt
        allow_interrupt = (priority == 0)

        return self.speak(text, priority=priority, allow_interrupt=allow_interrupt)

    def _enhance_template(
        self,
        template: str,
        state: dict[str, Any],
        movement_state: str
    ) -> str:
        """Enhance template with state information."""
        # Truncate for combat
        if movement_state == "combat" and len(template) > 50:
            # Extract first sentence
            sentences = template.split('.')[0]
            return sentences + "!" if not sentences.endswith('!') else sentences

        return template

    def interrupt(self) -> None:
        """Interrupt current speech."""
        self.interrupt_requested = True
        self.speech_state = SpeechState.INTERRUPTED

        # Cancel via WebSocket
        if self.loop and self.ws_connected:
            asyncio.run_coroutine_threadsafe(
                self._cancel_response(),
                self.loop
            )

    def stop(self) -> None:
        """Stop current speech and clear queue."""
        self.interrupt()
        self.speech_state = SpeechState.IDLE

        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

    async def initialize(self) -> bool:
        """
        Initialize the voice client.

        Returns:
            True if initialized successfully
        """
        if not self.enabled:
            return False

        if self.use_realtime_api:
            return await self._connect_realtime_api()

        return True

    def shutdown(self) -> None:
        """Shutdown client and cleanup resources."""
        self.stop()

        if self.loop and self.ws_connected:
            asyncio.run_coroutine_threadsafe(
                self._disconnect_realtime_api(),
                self.loop
            )

        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

        self.enabled = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


def create_voice_client(
    api_key: str | None = None,
    voice: str = "alloy",
    enable_audio: bool = True,
    use_realtime: bool = True
) -> RealtimeVoiceClient:
    """
    Create a Realtime Voice client with sensible defaults.

    Args:
        api_key: OpenAI API key
        voice: Voice to use
        enable_audio: Enable audio output
        use_realtime: Use Realtime API

    Returns:
        Configured RealtimeVoiceClient instance
    """
    return RealtimeVoiceClient(
        api_key=api_key,
        voice=voice,
        enable_audio_output=enable_audio,
        use_realtime_api=use_realtime,
        cooldown_ms=3000,
        max_response_length_ms=10000
    )
