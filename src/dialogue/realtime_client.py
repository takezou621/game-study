"""Realtime voice client using OpenAI Realtime API."""

import asyncio
import json
import time
import logging
from typing import Optional, Dict, Any, Callable, List
import numpy as np


# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import audio libraries
try:
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

from ..utils.rate_limiter import RateLimiter
from ..utils.exceptions import APIError
from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class RealtimeVoiceClient:
    """
    OpenAI Realtime API client for voice conversation.

    Supports audio input/output with low latency.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-realtime-preview",
        system_prompt_path: Optional[str] = None,
        voice: str = "alloy",
        rate_limit_calls: int = 30,
        rate_limit_period: float = 60.0,
    ):
        """
        Initialize Realtime voice client.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: Model to use
            system_prompt_path: Path to system prompt file
            voice: Voice to use (alloy, echo, etc.)
            rate_limit_calls: Maximum API calls per period
            rate_limit_period: Time period for rate limiting (seconds)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library is not available. "
                "Install with: pip install openai"
            )

        # Store API key for later use when connecting, but keep it internal
        self._api_key = api_key
        self.model = model
        self.voice = voice
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        # Audio settings
        self.sample_rate = 24000  # OpenAI Realtime API requirement
        self.channels = 1  # Mono

        # OpenAI client
        self.client: Optional[openai.AsyncOpenAI] = None
        self.realtime: Optional[openai.AsyncRealtime] = None

        # Rate limiter for API calls
        self.rate_limiter = RateLimiter(
            max_calls=rate_limit_calls,
            period_seconds=rate_limit_period,
            name=f"Realtime.{model}"
        )

        # Audio buffers
        self._audio_input_buffer: List[float] = []
        self._max_input_buffer = 4800  # 200ms at 24kHz

        # Callbacks
        self._on_audio_delta: Optional[Callable] = None
        self._on_text_delta: Optional[Callable] = None
        self._on_function_call: Optional[Callable] = None

        # State
        self._connected = False
        self._speaking = False
        self._interrupt_requested = False

        # Statistics
        self._latency_samples = []
        self._total_audio_sent = 0
        self._total_audio_received = 0

    def _load_system_prompt(self, path: Optional[str]) -> str:
        """
        Load system prompt from file.

        Args:
            path: Path to system prompt file

        Returns:
            System prompt string
        """
        if path:
            try:
                with open(path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error("Failed to load system prompt", exc_info=True)

        # Default system prompt
        return """You are an AI English teacher and gaming coach for Fortnite players. Your goal is to help players improve their English skills while playing the game naturally.

## Communication Style

- Keep responses SHORT during combat (1-2 sentences max)
- Use clear, natural gaming English
- When confidence is low, ask questions instead of making statements
- Teach vocabulary in context, not as a list

## Priority Guidelines

- **P0 (Survival)**: Urgent, direct commands
- **P1 (Tactical)**: Informative, strategic suggestions
- **P2 (Learning)**: Educational, contextual vocabulary lessons
- **P3 (Chatter)**: Casual conversation, session review

## Combat vs Non-Combat

- **Combat**: Maximum 2 sentences. Focus on survival.
- **Non-Combat**: 2-4 sentences. You can explain and teach.

Remember: The player is here to learn English while gaming. Keep it fun, practical, and supportive!
"""

    async def connect(self):
        """
        Connect to OpenAI Realtime API.

        Raises:
            RuntimeError if API key is not set
        """
        api_key = self._api_key or openai.api_key
        if not api_key:
            raise RuntimeError("OpenAI API key is required")

        # Initialize AsyncOpenAI client with api_key directly
        self.client = openai.AsyncOpenAI(api_key=api_key)

        # Connect to Realtime API
        self.realtime = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input_format="pcm16",
            output_format="pcm16",
            temperature=0.7,
        )

        self._connected = True
        logger.info(f"Connected to OpenAI Realtime API (model: {self.model})")

    async def disconnect(self):
        """Disconnect from OpenAI Realtime API."""
        if self.realtime:
            await self.realtime.close()
            self.realtime = None

        self._connected = False
        logger.info("Disconnected from OpenAI Realtime API")

    async def send_audio(self, audio_data: np.ndarray) -> bool:
        """
        Send audio data to OpenAI Realtime API.

        Args:
            audio_data: Audio data as numpy array (float32 or int16)

        Returns:
            True if sent successfully
        """
        if not self._connected or self.realtime is None:
            return False

        # Check rate limit
        if not self.rate_limiter.allow_call():
            wait_time = self.rate_limiter.wait_time()
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        try:
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Send audio through the realtime API
            await self.realtime.send_audio(audio_data.tolist())

            self._total_audio_sent += len(audio_data)

            # Update latency tracking
            latency_start = time.time()

            @self.realtime.on("audio_delta")
            def track_latency(delta):
                latency_ms = (time.time() - latency_start) * 1000
                self._latency_samples.append(latency_ms)

            return True

        except Exception as e:
            logger.error("Failed to send audio", exc_info=True)
            return False

    async def send_text(self, text: str) -> bool:
        """
        Send text message to OpenAI Realtime API.

        Args:
            text: Text message to send

        Returns:
            True if sent successfully
        """
        if not self._connected or self.realtime is None:
            return False

        # Check rate limit
        if not self.rate_limiter.allow_call():
            wait_time = self.rate_limiter.wait_time()
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        try:
            await self.realtime.send_text(text)
            return True

        except Exception as e:
            logger.error("Failed to send text", exc_info=True)
            return False

    def on_audio_delta(self, callback: Callable):
        """
        Set callback for audio delta events.

        Args:
            callback: Callback function taking audio data
        """
        self._on_audio_delta = callback

    def on_text_delta(self, callback: Callable):
        """
        Set callback for text delta events.

        Args:
            callback: Callback function taking text string
        """
        self._on_text_delta = callback

    def on_function_call(self, callback: Callable):
        """
        Set callback for function call events.

        Args:
            callback: Callback function taking function call data
        """
        self._on_function_call = callback

    async def interrupt(self):
        """
        Interrupt current speech.

        Stops current TTS output.
        """
        self._interrupt_requested = True
        logger.debug("Speech interrupted")

    def reset_interrupt(self):
        """Reset interrupt flag."""
        self._interrupt_requested = False

    def is_speaking(self) -> bool:
        """
        Check if AI is currently speaking.

        Returns:
            True if speaking
        """
        return self._speaking

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get realtime voice client statistics.

        Returns:
            Statistics dictionary
        """
        avg_latency = (
            sum(self._latency_samples) / len(self._latency_samples)
            if self._latency_samples else 0.0
        )

        return {
            "connected": self._connected,
            "model": self.model,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "speaking": self._speaking,
            "audio_sent_bytes": self._total_audio_sent,
            "audio_received_bytes": self._total_audio_received,
            "latency_ms_avg": avg_latency,
            "latency_samples": len(self._latency_samples),
            "interrupt_requested": self._interrupt_requested,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
