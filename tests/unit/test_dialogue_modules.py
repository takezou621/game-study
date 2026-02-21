"""Tests for dialogue modules (OpenAI client, Realtime client)."""

import os
import asyncio
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import pytest


# ============================================================================
# OpenAI Client Tests
# ============================================================================

class TestOpenAIClient:
    """Tests for OpenAI client."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        from dialogue.openai_client import OpenAIClient
        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            with pytest.raises(ValueError):
                OpenAIClient()
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="test-key")
        assert client is not None

    def test_generate_response_template_only(self):
        """Test response generation with template only."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="test-key")

        trigger_info = {
            'rule_id': 'test',
            'template': 'Low HP! Heal!'
        }
        state = {'player': {'status': {'hp': {'value': 20}}}}
        movement = 'combat'

        response = client.generate_response(trigger_info, state, movement)
        assert response is not None

    def test_set_system_prompt(self):
        """Test setting system prompt."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="test-key")

        client.set_system_prompt("You are a helpful assistant.")
        # Should not crash

    def test_reset_conversation(self):
        """Test resetting conversation."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="test-key")

        client.reset_conversation()
        # Should not crash


# ============================================================================
# Realtime Client Tests
# ============================================================================

class TestRealtimeVoiceClient:
    """Tests for realtime voice client."""

    def test_init_basic(self):
        """Test basic initialization."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(enable_audio_output=False)
        assert client is not None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(
            api_key="test-key",
            enable_audio_output=False
        )
        assert client is not None

    def test_voice_response_dataclass(self):
        """Test VoiceResponse dataclass."""
        from dialogue.realtime_client import VoiceResponse
        response = VoiceResponse(
            text="Test response",
            audio_data=b"fake_audio",
            duration_ms=1000,
            priority=2
        )
        assert response.text == "Test response"
        assert response.duration_ms == 1000

    def test_speak_returns_response(self):
        """Test speak method returns response."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(
            api_key="test-key",
            enable_audio_output=False
        )

        result = client.speak("Hello", priority=2)
        # Should return VoiceResponse even without audio
        assert result is not None

    def test_speak_with_trigger(self):
        """Test speak_with_trigger method."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(
            api_key="test-key",
            enable_audio_output=False
        )

        trigger_info = {
            'rule_id': 'low_hp',
            'priority': 0,
            'template': 'Low HP!'
        }
        state = {'player': {'status': {'hp': {'value': 20}}}}
        movement = 'combat'

        result = client.speak_with_trigger(trigger_info, state, movement)
        # May return None if cooldown or other conditions

    def test_interrupt(self):
        """Test interrupt method."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="test-key")
        client.interrupt()
        assert client.interrupt_requested == True

    def test_shutdown(self):
        """Test shutdown method."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="test-key")
        client.shutdown()
        assert client.enabled == False

    def test_combat_templates(self):
        """Test combat templates exist."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="test-key")

        assert hasattr(client, 'COMBAT_TEMPLATES')
        assert 0 in client.COMBAT_TEMPLATES  # P0 priority

    def test_get_short_response(self):
        """Test short response for combat."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="test-key")

        trigger_info = {'rule_id': 'low_hp', 'priority': 0}
        short = client._get_short_response(trigger_info, 'combat')
        # May return None if no matching template

    def test_enhance_template(self):
        """Test template enhancement."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="test-key")

        template = "This is a very long template that should be shortened for combat situations."
        state = {}
        enhanced = client._enhance_template(template, state, 'combat')
        assert enhanced is not None
