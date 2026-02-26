"""Use case for generating voice responses."""

from dataclasses import dataclass
from typing import Any, Protocol

from application.dto.response_dto import AudioResponseDTO, ResponseDTO
from application.dto.trigger_dto import TriggerResultDTO
from application.ports.llm_port import LLMConfig, LLMPort
from application.ports.tts_port import TTSConfig, TTSPort


class TemplateRenderer(Protocol):
    """Protocol for template rendering."""

    def render(self, template: str, context: dict) -> str:
        """Render template with context."""
        ...


@dataclass
class GenerateResponseUseCase:
    """Use case for generating voice responses."""

    llm_port: LLMPort | None = None
    tts_port: TTSPort | None = None
    template_renderer: TemplateRenderer | None = None
    default_voice: str = "alloy"
    max_response_chars: int = 200

    def execute_from_trigger(
        self,
        trigger: TriggerResultDTO,
        state: dict,
        use_llm: bool = False,
    ) -> ResponseDTO:
        """Generate response from trigger.

        Args:
            trigger: Trigger that fired
            state: Current game state for context
            use_llm: Whether to use LLM for generation

        Returns:
            ResponseDTO with generated text
        """
        if use_llm and self.llm_port is not None:
            text = self._generate_with_llm(trigger, state)
        else:
            text = self._generate_from_template(trigger, state)

        # Truncate if needed
        if len(text) > self.max_response_chars:
            text = text[: self.max_response_chars - 3] + "..."

        return ResponseDTO(
            text=text,
            priority=trigger.priority,
            source_trigger=trigger.trigger_id,
            movement_state=trigger.movement_state,
        )

    def execute_text_to_speech(
        self,
        response: ResponseDTO,
    ) -> AudioResponseDTO | None:
        """Convert text response to audio.

        Args:
            response: Text response to convert

        Returns:
            AudioResponseDTO or None if TTS unavailable
        """
        if self.tts_port is None:
            return None

        config = TTSConfig(
            voice=self.default_voice,
            speed=1.0 if response.priority >= 2 else 1.1,  # Slightly faster for urgent
        )

        try:
            result = self.tts_port.synthesize(response.text, config)
            return AudioResponseDTO(
                text=response.text,
                audio_data=result.audio_data,
                duration_ms=result.duration_ms,
                priority=response.priority,
                sample_rate=result.sample_rate,
                source_trigger=response.source_trigger,
            )
        except Exception:
            return None

    def _generate_with_llm(self, trigger: TriggerResultDTO, state: dict) -> str:
        """Generate response using LLM."""
        if self.llm_port is None:
            return trigger.template or ""

        prompt = self._build_llm_prompt(trigger, state)
        config = LLMConfig(
            max_tokens=100,
            temperature=0.7,
        )

        try:
            response = self.llm_port.complete(prompt, config)
            return response.content.strip()
        except Exception:
            return trigger.template or ""

    def _generate_from_template(
        self,
        trigger: TriggerResultDTO,
        state: dict,
    ) -> str:
        """Generate response from template."""
        template = trigger.template
        if template is None:
            return ""

        if self.template_renderer is not None:
            return self.template_renderer.render(template, state)

        return template

    def _build_llm_prompt(self, trigger: TriggerResultDTO, state: dict) -> str:
        """Build LLM prompt from trigger and state."""
        # Extract key values from state
        hp = self._get_state_value(state, "player.status.hp")
        shield = self._get_state_value(state, "player.status.shield")
        in_storm = self._get_state_value(state, "world.storm.in_storm")

        prompt = f"""You are a gaming coach giving brief advice.
Trigger: {trigger.trigger_name}
Situation: HP={hp}, Shield={shield}, In Storm={in_storm}
Priority: {trigger.priority} (0=highest/urgent, 3=lowest/casual)

Give a very brief (1-2 sentences) coaching tip. Be direct and actionable.
For priority 0-1 (urgent), keep it under 10 words.
For priority 2-3 (casual), keep it under 30 words.

Response:"""

        return prompt

    def _get_state_value(self, state: dict, path: str) -> Any:
        """Get value from state by path."""
        keys = path.split(".")
        current = state
        for key in keys:
            if isinstance(current, dict):
                if key in current:
                    val = current[key]
                    if isinstance(val, dict) and "value" in val:
                        current = val["value"]
                    else:
                        current = val
                else:
                    return None
            else:
                return None
        return current

    @classmethod
    def create(
        cls,
        llm_port: LLMPort | None = None,
        tts_port: TTSPort | None = None,
        template_renderer: TemplateRenderer | None = None,
    ) -> "GenerateResponseUseCase":
        """Factory method to create use case.

        Args:
            llm_port: Optional LLM port for AI generation
            tts_port: Optional TTS port for speech synthesis
            template_renderer: Optional template renderer

        Returns:
            Configured GenerateResponseUseCase
        """
        return cls(
            llm_port=llm_port,
            tts_port=tts_port,
            template_renderer=template_renderer,
        )
