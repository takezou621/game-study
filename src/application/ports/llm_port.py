"""LLM port interface for language model interactions."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    tokens_used: int
    finish_reason: str
    latency_ms: int


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    model: str = "gpt-4o"
    max_tokens: int = 500
    temperature: float = 0.7
    system_prompt: str | None = None


@runtime_checkable
class LLMPort(Protocol):
    """Port (interface) for LLM functionality."""

    def complete(self, prompt: str, config: LLMConfig | None = None) -> LLMResponse:
        """Generate completion for prompt.

        Args:
            prompt: User prompt
            config: Optional configuration override

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If generation fails
        """
        ...

    def complete_stream(self, prompt: str, config: LLMConfig | None = None):
        """Generate streaming completion for prompt.

        Args:
            prompt: User prompt
            config: Optional configuration override

        Yields:
            Chunks of generated content

        Raises:
            LLMError: If generation fails
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        ...


class RealtimeLLMPort(Protocol):
    """Port for realtime LLM interactions (e.g., OpenAI Realtime API)."""

    async def connect(self) -> None:
        """Connect to realtime service."""
        ...

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data for processing."""
        ...

    async def receive_audio(self) -> bytes:
        """Receive audio response.

        Returns:
            Audio data bytes
        """
        ...

    async def interrupt(self) -> None:
        """Interrupt current generation."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from realtime service."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to realtime service."""
        ...
