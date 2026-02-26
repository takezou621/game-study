"""WebSocket connection management for OpenAI Realtime API."""

import asyncio
import json
import os
from typing import Any

from infrastructure.exceptions import ConnectionError, LLMError
from utils.dependencies import WEBSOCKETS_AVAILABLE
from utils.logger import get_logger

if WEBSOCKETS_AVAILABLE:
    import websockets

logger = get_logger(__name__)


class RealtimeConnection:
    """
    Manages WebSocket connection to OpenAI Realtime API.

    Handles connection lifecycle, authentication, and session configuration.
    """

    DEFAULT_URL = "wss://api.openai.com/v1/realtime"
    DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-12-17"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = "alloy",
        system_prompt: str = "",
    ):
        """
        Initialize connection manager.

        Args:
            api_key: OpenAI API key
            model: Realtime model to use
            voice: Voice to use
            system_prompt: System instructions
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.system_prompt = system_prompt

        self._ws: Any | None = None
        self._connected = False

        logger.debug(
            "RealtimeConnection initialized",
            extra={"model": model, "voice": voice}
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to Realtime API."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to Realtime API.

        Returns:
            True if connected successfully

        Raises:
            ConnectionError: If connection fails
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets library not available")
            raise ConnectionError(
                service="OpenAI Realtime API",
                reason="WebSockets library not installed",
                retryable=False,
            )

        if not self.api_key:
            logger.error("No API key available for Realtime API")
            raise ConnectionError(
                service="OpenAI Realtime API",
                reason="API key not provided",
                retryable=False,
            )

        try:
            logger.info(f"Connecting to OpenAI Realtime API (model: {self.model})")

            headers = [
                ("Authorization", f"Bearer {self.api_key}"),
                ("OpenAI-Beta", "realtime=v1"),
            ]

            self._ws = await websockets.connect(
                self.DEFAULT_URL,
                extra_headers=headers,
            )

            # Configure session
            await self._configure_session()

            # Wait for session.created event
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            event = json.loads(response)

            if event.get("type") == "session.created":
                self._connected = True
                logger.info(
                    "Successfully connected to Realtime API",
                    extra={"session_id": event.get("session", {}).get("id")}
                )
                return True

            logger.warning(
                "Unexpected response during connection",
                extra={"event_type": event.get("type")}
            )
            return False

        except asyncio.TimeoutError:
            logger.error("Timeout waiting for session creation from Realtime API")
            raise ConnectionError(
                service="OpenAI Realtime API",
                reason="Timeout waiting for session creation",
                retryable=True,
                retry_after_ms=5000,
            )
        except ConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            raise ConnectionError(
                service="OpenAI Realtime API",
                reason=str(e),
                cause=e,
            )

    async def _configure_session(self) -> None:
        """Send session configuration."""
        config = {
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
                    "silence_duration_ms": 500,
                },
            },
        }

        logger.debug("Sending session configuration to Realtime API")
        await self._ws.send(json.dumps(config))

    async def disconnect(self) -> None:
        """Disconnect from Realtime API."""
        if self._ws:
            try:
                await self._ws.close()
                logger.info("Disconnected from Realtime API")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._ws = None
                self._connected = False

    async def send(self, message: dict[str, Any]) -> None:
        """
        Send message to API.

        Args:
            message: Message dictionary

        Raises:
            LLMError: If not connected or send fails
        """
        if not self._connected or not self._ws:
            logger.error("Attempted to send message while not connected")
            raise LLMError(
                provider="openai",
                operation="realtime_send",
                reason="Not connected to Realtime API",
            )

        try:
            await self._ws.send(json.dumps(message))
            logger.debug(f"Sent message to Realtime API: {message.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to send message to Realtime API: {e}")
            raise LLMError(
                provider="openai",
                operation="realtime_send",
                reason=str(e),
                cause=e,
            )

    async def receive(self, timeout: float = 30.0) -> dict[str, Any] | None:
        """
        Receive message from API.

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Message dictionary or None on timeout

        Raises:
            LLMError: If not connected or receive fails
        """
        if not self._connected or not self._ws:
            logger.error("Attempted to receive message while not connected")
            raise LLMError(
                provider="openai",
                operation="realtime_receive",
                reason="Not connected to Realtime API",
            )

        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            message = json.loads(raw)
            logger.debug(f"Received message from Realtime API: {message.get('type', 'unknown')}")
            return message
        except asyncio.TimeoutError:
            logger.debug("Receive timeout - no message available")
            return None
        except Exception as e:
            logger.error(f"Failed to receive message from Realtime API: {e}")
            raise LLMError(
                provider="openai",
                operation="realtime_receive",
                reason=str(e),
                cause=e,
            )

    async def cancel_response(self) -> None:
        """Cancel current response generation."""
        if self._connected and self._ws:
            try:
                await self.send({"type": "response.cancel"})
                logger.debug("Cancelled response generation")
            except Exception as e:
                logger.warning(f"Error cancelling response: {e}")
