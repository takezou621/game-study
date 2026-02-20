"""OpenAI API client for generating AI coach responses."""

import os
import time
import logging
from typing import Dict, Any, Optional, List

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..utils.rate_limiter import RateLimiter
from ..utils.exceptions import APIError, RateLimitError
from ..utils.retry import retry_with_backoff_sync

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client for generating AI coach responses."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        system_prompt_path: Optional[str] = None,
        rate_limit_calls: int = 60,
        rate_limit_period: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: Model to use
            system_prompt_path: Path to system prompt file
            rate_limit_calls: Maximum API calls per period
            rate_limit_period: Time period for rate limiting (seconds)
            max_retries: Maximum number of retry attempts for API calls
        """
        if not OPENAI_AVAILABLE:
            self.model = model
            self.client = None
            self.rate_limiter = None
            self.max_retries = max_retries
            self.system_prompt = self._load_system_prompt(system_prompt_path)
            self.conversation_history: List[Dict[str, str]] = []
            return

        # Get API key from parameter or environment, but don't store it
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        # Pass api_key directly to OpenAI client, don't store in self
        self.client = OpenAI(api_key=key)
        self.rate_limiter = RateLimiter(
            max_calls=rate_limit_calls,
            period_seconds=rate_limit_period,
            name=f"OpenAI.{model}"
        )
        self.max_retries = max_retries
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.conversation_history: List[Dict[str, str]] = []

        logger.info(
            f"OpenAIClient initialized: model={model}, "
            f"rate_limit={rate_limit_calls}/{rate_limit_period}s, "
            f"max_retries={max_retries}"
        )

    def _load_system_prompt(self, path: Optional[str]) -> str:
        """
        Load system prompt from file.

        Args:
            path: Path to system prompt file

        Returns:
            System prompt string
        """
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()

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
"""

    def generate_response(
        self,
        trigger_info: Dict[str, Any],
        state: Dict[str, Any],
        movement_state: str,
        max_length: int = 200
    ) -> str:
        """
        Generate AI response based on trigger and state.

        Args:
            trigger_info: Trigger information (id, name, priority, template)
            state: Current game state
            movement_state: Movement state ("combat" or "non_combat")
            max_length: Maximum response length in characters

        Returns:
            Generated response string
        """
        # Use template if available (MVP behavior)
        template = trigger_info.get('template')
        if template:
            # For MVP, use template directly
            # In Phase 2+, we can enhance with OpenAI
            return self._enhance_template(template, state, movement_state, max_length)

        # If no template, generate response with OpenAI
        return self._generate_with_openai(trigger_info, state, movement_state, max_length)

    def _enhance_template(
        self,
        template: str,
        state: Dict[str, Any],
        movement_state: str,
        max_length: int
    ) -> str:
        """
        Enhance template with state information.

        Args:
            template: Base template
            state: Game state
            movement_state: Movement state
            max_length: Maximum length

        Returns:
            Enhanced template
        """
        # MVP: Return template as-is
        # In Phase 2+, we can add context from state
        return template[:max_length]

    def _generate_with_openai(
        self,
        trigger_info: Dict[str, Any],
        state: Dict[str, Any],
        movement_state: str,
        max_length: int
    ) -> str:
        """
        Generate response using OpenAI API.

        Args:
            trigger_info: Trigger information
            state: Game state
            movement_state: Movement state
            max_length: Maximum length in characters

        Returns:
            Generated response
        """
        if not OPENAI_AVAILABLE or self.client is None:
            # Fallback to default response when OpenAI is not available
            return f"Response: {trigger_info.get('rule_name', 'Unknown')}"

        # Build context
        context = self._build_context(trigger_info, state, movement_state)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context},
        ]

        # Make API call with retry and rate limiting
        def _make_api_call():
            # Check rate limit first
            if self.rate_limiter and not self.rate_limiter.allow_call():
                wait_time = self.rate_limiter.wait_time()
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100,
                temperature=0.7,
            )

            response_text = response.choices[0].message.content.strip()

            # Truncate to max length
            if len(response_text) > max_length:
                response_text = response_text[:max_length].rsplit(' ', 1)[0] + '...'

            return response_text

        try:
            # Use retry with backoff if max_retries > 0
            if self.max_retries > 0:
                return retry_with_backoff_sync(
                    _make_api_call,
                    max_retries=self.max_retries,
                    base_delay=1.0,
                    max_delay=60.0,
                    retryable_exceptions=(
                        APIError,
                        RateLimitError,
                        ConnectionError,
                        TimeoutError,
                    ),
                )
            else:
                # No retry, just call directly
                return _make_api_call()

        except Exception as e:
            # Log full exception details with exc_info=True
            logger.error("Response generation failed", exc_info=True)
            # Fallback response on error (sanitized, no sensitive info)
            return "Response generation failed. Please try again."

    def _build_context(
        self,
        trigger_info: Dict[str, Any],
        state: Dict[str, Any],
        movement_state: str
    ) -> str:
        """
        Build context string for OpenAI.

        Args:
            trigger_info: Trigger information
            state: Game state
            movement_state: Movement state

        Returns:
            Context string
        """
        context_parts = [
            f"Trigger: {trigger_info.get('name')} (Priority {trigger_info.get('priority')})",
            f"Movement State: {movement_state}",
        ]

        # Add relevant state information
        hp = state["player"]["status"]["hp"]["value"]
        shield = state["player"]["status"]["shield"]["value"]
        if hp is not None:
            context_parts.append(f"HP: {hp}")
        if shield is not None:
            context_parts.append(f"Shield: {shield}")

        return "\n".join(context_parts)

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set system prompt.

        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
