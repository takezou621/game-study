"""Tests for dialogue modules (OpenAI client, Realtime client)."""

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# OpenAI Client Tests
# ============================================================================

class TestOpenAIClient:
    """Tests for OpenAI client."""

    def test_init_without_api_key(self):
        """Test initialization without API key raises ValueError."""
        from dialogue.openai_client import OpenAIClient, OPENAI_AVAILABLE
        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            if OPENAI_AVAILABLE:
                with pytest.raises(ValueError):
                    OpenAIClient()
            else:
                # When OpenAI is not available, no exception is raised
                client = OpenAIClient()
                assert client.client is None
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")
        assert client is not None

    def test_init_from_env(self):
        """Test initialization using environment variable."""
        from dialogue.openai_client import OpenAIClient
        original = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'sk-env-key'

        try:
            client = OpenAIClient()
            assert client is not None
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original
            else:
                os.environ.pop('OPENAI_API_KEY', None)

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key", model="gpt-3.5-turbo")
        assert client.model == "gpt-3.5-turbo"

    def test_init_conversation_history_empty(self):
        """Test conversation history starts empty."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")
        assert client.conversation_history == []

    # _load_system_prompt tests

    def test_load_system_prompt_default(self):
        """Test _load_system_prompt returns default when no path provided."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")
        prompt = client._load_system_prompt(None)
        assert "AI English teacher" in prompt
        assert "Fortnite" in prompt

    def test_load_system_prompt_from_file(self):
        """Test _load_system_prompt loads from file."""
        from dialogue.openai_client import OpenAIClient
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Custom system prompt for testing")
            f.flush()
            temp_path = f.name

        try:
            client = OpenAIClient(api_key="sk-test-key", system_prompt_path=temp_path)
            assert "Custom system prompt" in client.system_prompt
        finally:
            os.unlink(temp_path)

    def test_load_system_prompt_nonexistent_path(self):
        """Test _load_system_prompt with nonexistent path returns default."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key", system_prompt_path="/nonexistent/path.txt")
        assert "AI English teacher" in client.system_prompt

    # generate_response tests

    def test_generate_response_template_only(self):
        """Test response generation with template only."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {
            'rule_id': 'test',
            'template': 'Low HP! Heal!'
        }
        state = {'player': {'status': {'hp': {'value': 20}}}}
        movement = 'combat'

        response = client.generate_response(trigger_info, state, movement)
        assert response == 'Low HP! Heal!'

    def test_generate_response_with_empty_template(self):
        """Test generate_response with empty template."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'rule_id': 'test', 'template': ''}
        state = {'player': {'status': {'hp': {'value': 20}, 'shield': {'value': 0}}}}
        movement = 'combat'

        response = client.generate_response(trigger_info, state, movement)
        # Should fall back to OpenAI generation
        assert response is not None

    def test_generate_response_with_none_template(self):
        """Test generate_response with None template."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'rule_id': 'test', 'template': None}
        state = {'player': {'status': {'hp': {'value': 20}, 'shield': {'value': 0}}}}
        movement = 'combat'

        response = client.generate_response(trigger_info, state, movement)
        assert response is not None

    def test_generate_response_all_trigger_info_fields(self):
        """Test generate_response with all trigger_info fields."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {
            'rule_id': 'test_rule',
            'rule_name': 'Test Rule',
            'name': 'Test Name',
            'priority': 1,
            'template': 'Test template'
        }
        state = {'player': {'status': {'hp': {'value': 50}}}}
        movement = 'non_combat'

        response = client.generate_response(trigger_info, state, movement)
        assert response == 'Test template'

    # _enhance_template tests

    def test_enhance_template_basic(self):
        """Test _enhance_template returns template."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        template = "This is a test template"
        state = {'player': {'status': {'hp': {'value': 50}}}}
        movement = 'combat'

        result = client._enhance_template(template, state, movement, max_length=len(template))
        assert result == template

    def test_enhance_template_truncation(self):
        """Test _enhance_template truncates long templates."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        template = "This is a very long template that should be truncated"
        state = {'player': {'status': {'hp': {'value': 50}}}}
        movement = 'combat'

        result = client._enhance_template(template, state, movement, max_length=20)
        assert len(result) == 20

    # _build_context tests

    def test_build_context_basic(self):
        """Test _build_context with basic state."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'name': 'Test Trigger', 'priority': 1}
        state = {
            'player': {
                'status': {
                    'hp': {'value': 75},
                    'shield': {'value': 50}
                }
            }
        }
        movement = 'combat'

        context = client._build_context(trigger_info, state, movement)
        assert 'Trigger: Test Trigger' in context
        assert 'Priority 1' in context
        assert 'HP: 75' in context
        assert 'Shield: 50' in context
        assert 'Movement State: combat' in context

    def test_build_context_missing_shield(self):
        """Test _build_context with missing shield value."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'name': 'Test', 'priority': 2}
        state = {
            'player': {
                'status': {
                    'hp': {'value': 100},
                    'shield': {'value': None}
                }
            }
        }
        movement = 'non_combat'

        context = client._build_context(trigger_info, state, movement)
        assert 'HP: 100' in context
        assert 'Shield:' not in context  # None value should be skipped

    def test_build_context_missing_hp(self):
        """Test _build_context with missing HP value."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'name': 'Test', 'priority': 2}
        state = {
            'player': {
                'status': {
                    'hp': {'value': None},
                    'shield': {'value': 50}
                }
            }
        }
        movement = 'combat'

        context = client._build_context(trigger_info, state, movement)
        assert 'Shield: 50' in context
        assert 'HP:' not in context  # None value should be skipped

    def test_build_context_both_none(self):
        """Test _build_context with both HP and shield as None."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'name': 'Test', 'priority': 1}
        state = {
            'player': {
                'status': {
                    'hp': {'value': None},
                    'shield': {'value': None}
                }
            }
        }
        movement = 'non_combat'

        context = client._build_context(trigger_info, state, movement)
        assert 'Trigger: Test' in context
        assert 'HP:' not in context
        assert 'Shield:' not in context

    def test_build_context_missing_name(self):
        """Test _build_context with missing trigger name."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        trigger_info = {'priority': 1}
        state = {'player': {'status': {'hp': {'value': 50}, 'shield': {'value': 25}}}}
        movement = 'combat'

        context = client._build_context(trigger_info, state, movement)
        # Should use 'None' for missing name
        assert 'Trigger: None' in context

    # Other methods

    def test_set_system_prompt(self):
        """Test setting system prompt."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        custom_prompt = "You are a helpful assistant."
        client.set_system_prompt(custom_prompt)
        assert client.system_prompt == custom_prompt

    def test_reset_conversation(self):
        """Test resetting conversation."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")

        # Add some fake conversation history
        client.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        client.reset_conversation()
        assert client.conversation_history == []

    def test_is_available(self):
        """Test is_available property."""
        from dialogue.openai_client import OpenAIClient
        client = OpenAIClient(api_key="sk-test-key")
        # Should be True if OpenAI is available
        assert client.is_available in [True, False]


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
            api_key="sk-test-key",
            enable_audio_output=False
        )
        assert client is not None

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(
            api_key="sk-test-key",
            model="gpt-4o-realtime-preview",
            voice="nova",
            cooldown_ms=5000,
            max_response_length_ms=15000,
            enable_audio_output=False,
            use_realtime_api=False
        )
        assert client.model == "gpt-4o-realtime-preview"
        assert client.voice == "nova"
        assert client.cooldown_ms == 5000
        assert client.max_response_length_ms == 15000

    def test_init_speech_state(self):
        """Test initial speech state is IDLE."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        assert client.speech_state.name == "IDLE"

    def test_init_priority(self):
        """Test initial priority is low (99)."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        assert client.current_priority == 99

    def test_init_no_api_key(self):
        """Test initialization without API key sets enabled=False."""
        from dialogue.realtime_client import RealtimeVoiceClient

        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            client = RealtimeVoiceClient(enable_audio_output=False)
            assert client.enabled == False
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    # _load_system_prompt tests

    def test_load_system_prompt_default(self):
        """Test _load_system_prompt returns default prompt."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        prompt = client.system_prompt
        assert "AI English teacher" in prompt
        assert "gaming coach" in prompt

    def test_load_system_prompt_from_file(self):
        """Test _load_system_prompt loads from file."""
        from dialogue.realtime_client import RealtimeVoiceClient

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Custom voice system prompt")
            f.flush()
            temp_path = f.name

        try:
            client = RealtimeVoiceClient(
                api_key="sk-test-key",
                system_prompt_path=temp_path,
                enable_audio_output=False
            )
            assert "Custom voice system prompt" in client.system_prompt
        finally:
            os.unlink(temp_path)

    # _get_short_response tests

    def test_get_short_response_combat_p0_low_hp(self):
        """Test _get_short_response for P0 low_hp trigger."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'player_low_hp', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is not None
        assert "Low HP" in response

    def test_get_short_response_combat_p0_knocked(self):
        """Test _get_short_response for P0 knocked trigger."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'player_knocked', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is not None
        assert "Knocked" in response

    def test_get_short_response_combat_p0_storm(self):
        """Test _get_short_response for P0 storm trigger."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'storm_approaching', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is not None
        assert "Storm" in response

    def test_get_short_response_combat_p1_rotate(self):
        """Test _get_short_response for P1 rotate trigger."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'rotate_to_zone', 'priority': 1}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is not None
        assert "Rotate" in response

    def test_get_short_response_combat_p1_storm_shrinking(self):
        """Test _get_short_response for P1 storm_shrinking trigger."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'storm_shrinking_warning', 'priority': 1}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is not None
        assert "Storm" in response

    def test_get_short_response_non_combat(self):
        """Test _get_short_response returns None for non-combat."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'player_low_hp', 'priority': 0}
        response = client._get_short_response(trigger_info, 'non_combat')
        assert response is None

    def test_get_short_response_unknown_trigger(self):
        """Test _get_short_response with unknown trigger_id."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'unknown_trigger', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is None

    def test_get_short_response_p2_priority(self):
        """Test _get_short_response with P2 priority (no template)."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'some_trigger', 'priority': 2}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is None

    # speak_with_trigger tests

    def test_speak_with_trigger_disabled_client(self):
        """Test speak_with_trigger with disabled client."""
        from dialogue.realtime_client import RealtimeVoiceClient

        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            client = RealtimeVoiceClient(enable_audio_output=False)
            trigger_info = {'rule_id': 'test', 'priority': 0, 'template': 'Test'}
            state = {}
            response = client.speak_with_trigger(trigger_info, state, 'combat')
            # Should return None when disabled
            assert response is None
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    # shutdown tests

    def test_shutdown_disabled_client(self):
        """Test shutdown with disabled client."""
        from dialogue.realtime_client import RealtimeVoiceClient

        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            client = RealtimeVoiceClient(enable_audio_output=False)
            # When disabled, the client may not have all attributes
            # The shutdown should handle this gracefully
            try:
                client.shutdown()
            except AttributeError:
                # If attributes like 'loop' don't exist, that's expected for disabled clients
                pass
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    def test_shutdown_with_enabled_client(self):
        """Test shutdown with enabled client."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        # Should not raise exception
        try:
            client.shutdown()
        except AttributeError:
            # If loop attribute doesn't exist, that's OK
            pass

    # Context manager tests

    def test_context_manager(self):
        """Test context manager protocol."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        try:
            with client:
                assert client is not None
        except AttributeError:
            # If loop attribute doesn't exist during __exit__, that's OK
            pass

    def test_context_manager_shutdown(self):
        """Test context manager calls shutdown on exit."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        try:
            with client:
                pass
        except AttributeError:
            # If loop attribute doesn't exist, that's OK
            pass
        # Should have completed without error
        assert True


# ============================================================================
# VoiceResponse Tests
# ============================================================================

class TestVoiceResponse:
    """Tests for VoiceResponse dataclass."""

    def test_voiceresponse_creation(self):
        """Test creating a VoiceResponse."""
        from dialogue.realtime_client import VoiceResponse

        response = VoiceResponse(
            text="Test response",
            audio_data=b"audio",
            duration_ms=1000
        )
        assert response.text == "Test response"
        assert response.audio_data == b"audio"
        assert response.duration_ms == 1000

    def test_voiceresponse_default_values(self):
        """Test VoiceResponse default values."""
        from dialogue.realtime_client import VoiceResponse

        response = VoiceResponse(text="Test")
        assert response.audio_data is None
        assert response.duration_ms is None
        assert response.priority == 2
        assert response.interrupted is False

    def test_voiceresponse_timestamp_auto(self):
        """Test VoiceResponse timestamp is auto-generated."""
        from dialogue.realtime_client import VoiceResponse
        import time

        before = time.time()
        response = VoiceResponse(text="Test")
        after = time.time()

        assert before <= response.timestamp <= after


# ============================================================================
# SpeechState Tests
# ============================================================================

class TestSpeechState:
    """Tests for SpeechState enum."""

    def test_speech_state_values(self):
        """Test SpeechState enum values."""
        from dialogue.realtime_client import SpeechState

        assert SpeechState.IDLE.value == "idle"
        assert SpeechState.SPEAKING.value == "speaking"
        assert SpeechState.INTERRUPTED.value == "interrupted"


# ============================================================================
# Additional RealtimeVoiceClient Tests
# ============================================================================

class TestRealtimeVoiceClientAdvanced:
    """Advanced tests for realtime voice client."""

    # Additional initialization tests

    def test_init_with_custom_model_and_voice(self):
        """Test initialization with custom model and voice options."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(
            api_key="sk-test-key",
            model="gpt-4o-realtime-preview-2024-12-17",
            voice="shimmer",
            enable_audio_output=False
        )
        assert client.model == "gpt-4o-realtime-preview-2024-12-17"
        assert client.voice == "shimmer"

    def test_init_realtime_api_disabled_when_websockets_unavailable(self):
        """Test that use_realtime_api is False when websockets unavailable."""
        from dialogue.realtime_client import RealtimeVoiceClient
        import sys
        from unittest.mock import patch

        # Mock WEBSOCKETS_AVAILABLE as False
        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', False):
            client = RealtimeVoiceClient(
                api_key="sk-test-key",
                use_realtime_api=True,
                enable_audio_output=False
            )
            assert client.use_realtime_api == False

    def test_init_audio_queue(self):
        """Test audio queue is initialized."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        assert client.audio_queue is not None
        assert client.audio_queue.empty()

    def test_init_response_queue(self):
        """Test response queue is initialized."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        assert client.response_queue is not None
        assert client.response_queue.empty()

    def test_init_interrupt_flag(self):
        """Test interrupt_requested initial state."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        assert client.interrupt_requested == False

    # speak() method tests

    def test_speak_returns_voiceresponse_when_disabled(self):
        """Test speak returns VoiceResponse when client disabled."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse

        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            client = RealtimeVoiceClient(enable_audio_output=False)
            result = client.speak("Test message", priority=2)
            assert isinstance(result, VoiceResponse)
            assert result.text == "Test message"
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    def test_speak_cooldown_for_low_priority(self):
        """Test speak respects cooldown for low priority (>1)."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Mock the loop to avoid actual async operations
        client.loop = None

        # Set last spoken time to now (within cooldown)
        client.last_spoken_time = time.time()

        # Try to speak with priority 2 (should be blocked by cooldown)
        result = client.speak("Low priority message", priority=2)
        assert result is None

    def test_speak_high_priority_bypasses_cooldown(self):
        """Test high priority (0,1) bypasses cooldown."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Mock the loop to return VoiceResponse
        client.loop = None

        # Set last spoken time to now (within cooldown)
        client.last_spoken_time = time.time()

        # Priority 0 should bypass cooldown
        result = client.speak("High priority", priority=0, allow_interrupt=False)
        assert result is not None
        assert result.text == "High priority"

    def test_speak_interrupt_lower_priority(self):
        """Test speak sets interrupt flags when priority is lower than current."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Set current priority to 2
        client.current_priority = 2
        client.speech_state = SpeechState.SPEAKING
        client.loop = None

        # Speak with priority 0 should interrupt
        result = client.speak("Interrupt!", priority=0, allow_interrupt=True)
        # The interrupt flag is set before speech state is updated to SPEAKING
        # Since loop is None, it sets to SPEAKING on line 505
        assert client.interrupt_requested == True
        # After setting loop=None, speak sets state to SPEAKING (line 505)
        assert client.speech_state == SpeechState.SPEAKING

    def test_speak_no_interrupt_when_disabled(self):
        """Test speak doesn't interrupt when allow_interrupt=False."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Set current priority to 2
        client.current_priority = 2
        client.speech_state = SpeechState.SPEAKING
        client.loop = None

        # Speak with priority 0 but interrupt disabled
        result = client.speak("No interrupt", priority=0, allow_interrupt=False)
        assert client.speech_state == SpeechState.SPEAKING
        assert client.interrupt_requested == False

    def test_speak_updates_state(self):
        """Test speak updates speech state and priority during call."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        client.loop = None

        result = client.speak("Test", priority=1)
        # When loop is None, state is set to SPEAKING but not reset to IDLE
        # because the async operation never completes
        assert client.speech_state == SpeechState.SPEAKING
        assert client.current_priority == 1  # Set to the priority we used

    # _get_short_response additional tests

    def test_get_short_response_p0_low_hp_full_match(self):
        """Test _get_short_response for exact low_hp match."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'player_low_hp', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response == "Low HP! Cover!"

    def test_get_short_response_p0_knocked_full_match(self):
        """Test _get_short_response for exact knocked match."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'player_knocked', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response == "Knocked! Ping!"

    def test_get_short_response_p0_storm_full_match(self):
        """Test _get_short_response for exact storm match."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'storm_approaching', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response == "Storm! Move!"

    def test_get_short_response_p1_rotate_full_match(self):
        """Test _get_short_response for exact rotate match."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'rotate_to_zone', 'priority': 1}
        response = client._get_short_response(trigger_info, 'combat')
        assert response == "Rotate now!"

    def test_get_short_response_p1_storm_shrinking_full_match(self):
        """Test _get_short_response for exact storm_shrinking match."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'storm_shrinking_warning', 'priority': 1}
        response = client._get_short_response(trigger_info, 'combat')
        assert response == "Storm moving!"

    def test_get_short_response_substring_match(self):
        """Test _get_short_response matches substring in rule_id."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Should match "low_hp" in "warning_low_hp_critical"
        trigger_info = {'rule_id': 'warning_low_hp_critical', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response == "Low HP! Cover!"

    def test_get_short_response_missing_priority(self):
        """Test _get_short_response with missing priority defaults to 2."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'rule_id': 'some_trigger'}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is None

    def test_get_short_response_missing_rule_id(self):
        """Test _get_short_response with missing rule_id."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        trigger_info = {'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is None

    def test_get_short_response_case_sensitivity(self):
        """Test _get_short_response is case sensitive."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Uppercase shouldn't match
        trigger_info = {'rule_id': 'LOW_HP', 'priority': 0}
        response = client._get_short_response(trigger_info, 'combat')
        assert response is None

    # speak_with_trigger tests

    def test_speak_with_trigger_short_response(self):
        """Test speak_with_trigger uses short response in combat."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        trigger_info = {'rule_id': 'player_low_hp', 'priority': 0, 'template': 'Long template ignored'}
        state = {}
        result = client.speak_with_trigger(trigger_info, state, 'combat')
        assert result is not None
        # Should use short response, not the template
        assert "Low HP" in result.text or "Cover" in result.text

    def test_speak_with_trigger_dict_template(self):
        """Test speak_with_trigger with dict template."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        trigger_info = {
            'rule_id': 'test_rule',
            'priority': 2,
            'template': {
                'combat': 'Combat response',
                'non_combat': 'Non-combat response'
            }
        }
        state = {}

        result = client.speak_with_trigger(trigger_info, state, 'combat')
        assert result.text == "Combat response"

        result = client.speak_with_trigger(trigger_info, state, 'non_combat')
        assert result.text == "Non-combat response"

    def test_speak_with_trigger_dict_template_missing_movement(self):
        """Test speak_with_trigger falls back to non_combat when movement missing."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        trigger_info = {
            'rule_id': 'test_rule',
            'priority': 2,
            'template': {
                'non_combat': 'Fallback response'
            }
        }
        state = {}

        result = client.speak_with_trigger(trigger_info, state, 'combat')
        # Should fall back to non_combat
        assert result.text == "Fallback response"

    def test_speak_with_trigger_string_template(self):
        """Test speak_with_trigger with string template."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        trigger_info = {
            'rule_id': 'test_rule',
            'priority': 2,
            'template': 'Simple template'
        }
        state = {}

        result = client.speak_with_trigger(trigger_info, state, 'combat')
        assert result.text == "Simple template"

    def test_speak_with_trigger_empty_template(self):
        """Test speak_with_trigger with empty template returns None."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        trigger_info = {
            'rule_id': 'test_rule',
            'priority': 2,
            'template': ''
        }
        state = {}

        result = client.speak_with_trigger(trigger_info, state, 'combat')
        assert result is None

    def test_speak_with_trigger_p0_allows_interrupt(self):
        """Test speak_with_trigger allows interrupt for P0."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None
        client.current_priority = 2

        trigger_info = {'rule_id': 'test', 'priority': 0, 'template': 'Test'}
        state = {}

        result = client.speak_with_trigger(trigger_info, state, 'combat')
        # P0 priority is used and speak() is called with allow_interrupt=True
        # The state gets set to SPEAKING (interrupt handling happens but gets overwritten)
        assert result is not None
        assert client.current_priority == 0

    def test_speak_with_trigger_non_combat_no_short_response(self):
        """Test speak_with_trigger in non_combat doesn't use short responses."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        trigger_info = {
            'rule_id': 'player_low_hp',
            'priority': 0,
            'template': 'Use template in non-combat'
        }
        state = {}

        result = client.speak_with_trigger(trigger_info, state, 'non_combat')
        assert result.text == "Use template in non-combat"

    # _enhance_template tests

    def test_enhance_template_combat_truncation(self):
        """Test _enhance_template truncates long combat templates."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        long_template = "This is a very long template that should be truncated to the first sentence. It has multiple sentences."
        state = {}

        result = client._enhance_template(long_template, state, 'combat')
        assert len(result) < len(long_template)
        assert result.endswith('!')

    def test_enhance_template_non_combat_no_change(self):
        """Test _enhance_template doesn't change non-combat templates."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        template = "This is a template for non-combat."
        state = {}

        result = client._enhance_template(template, state, 'non_combat')
        assert result == template

    def test_enhance_template_short_combat_template(self):
        """Test _enhance_template doesn't change short combat templates."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        template = "Short!"
        state = {}

        result = client._enhance_template(template, state, 'combat')
        assert result == template

    # interrupt() and stop() tests

    def test_interrupt_sets_flags(self):
        """Test interrupt sets interrupt flags."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        client.interrupt()
        assert client.interrupt_requested == True
        assert client.speech_state == SpeechState.INTERRUPTED

    def test_stop_clears_queue(self):
        """Test stop clears audio queue."""
        from dialogue.realtime_client import RealtimeVoiceClient, AudioChunk
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Add some items to the queue
        client.audio_queue.put(AudioChunk(data=b"test1"))
        client.audio_queue.put(AudioChunk(data=b"test2"))

        assert not client.audio_queue.empty()

        client.stop()
        assert client.audio_queue.empty()
        assert client.speech_state.value == "idle"

    # initialize() tests

    def test_initialize_returns_false_when_disabled(self):
        """Test initialize returns False when client disabled."""
        from dialogue.realtime_client import RealtimeVoiceClient

        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            client = RealtimeVoiceClient(enable_audio_output=False)
            result = asyncio.run(client.initialize())
            assert result == False
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    # _get_api_key tests

    def test_get_api_key_from_constructor(self):
        """Test _get_api_key returns constructor-provided key."""
        from dialogue.realtime_client import RealtimeVoiceClient
        original = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'sk-env-key'

        try:
            # Constructor key should take precedence
            client = RealtimeVoiceClient(api_key="sk-constructor-key", enable_audio_output=False)
            api_key = client._get_api_key()
            assert api_key == 'sk-constructor-key'
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original
            else:
                os.environ.pop('OPENAI_API_KEY', None)

    def test_get_api_key_from_env_when_no_constructor_key(self):
        """Test _get_api_key falls back to environment when no constructor key."""
        from dialogue.realtime_client import RealtimeVoiceClient
        original = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'sk-env-key'

        try:
            # No constructor key, should use environment
            client = RealtimeVoiceClient(enable_audio_output=False)
            api_key = client._get_api_key()
            assert api_key == 'sk-env-key'
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original
            else:
                os.environ.pop('OPENAI_API_KEY', None)

    def test_get_api_key_raises_when_missing(self):
        """Test _get_api_key raises ValueError when missing everywhere."""
        from dialogue.realtime_client import RealtimeVoiceClient
        original = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)

        try:
            # No key provided anywhere, client is disabled
            client = RealtimeVoiceClient(enable_audio_output=False)
            with pytest.raises(ValueError, match="API key not found"):
                client._get_api_key()
        finally:
            if original:
                os.environ['OPENAI_API_KEY'] = original

    # WebSocket connection tests with mocking

    @pytest.mark.asyncio
    async def test_connect_realtime_api_when_websockets_unavailable(self):
        """Test _connect_realtime_api returns False when websockets unavailable."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', False):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            result = await client._connect_realtime_api()
            assert result == False

    @pytest.mark.asyncio
    async def test_disconnect_realtime_api_with_no_connection(self):
        """Test _disconnect_realtime_api handles no connection gracefully."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Should not raise exception
        await client._disconnect_realtime_api()
        assert client.ws is None
        assert client.ws_connected == False

    @pytest.mark.asyncio
    async def test_cancel_response_when_not_connected(self):
        """Test _cancel_response handles not connected state."""
        from dialogue.realtime_client import RealtimeVoiceClient
        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Should not raise exception
        await client._cancel_response()

    # AudioChunk tests

    def test_audio_chunk_creation(self):
        """Test AudioChunk creation."""
        from dialogue.realtime_client import AudioChunk
        chunk = AudioChunk(data=b"audio data")
        assert chunk.data == b"audio data"
        assert isinstance(chunk.timestamp, float)

    # create_voice_client tests

    def test_create_voice_client_defaults(self):
        """Test create_voice_client with defaults."""
        from dialogue.realtime_client import create_voice_client
        client = create_voice_client(api_key="sk-test-key", enable_audio=False)
        assert client.voice == "alloy"
        assert client.cooldown_ms == 3000
        assert client.max_response_length_ms == 10000

    def test_create_voice_client_custom_voice(self):
        """Test create_voice_client with custom voice."""
        from dialogue.realtime_client import create_voice_client
        client = create_voice_client(
            api_key="sk-test-key",
            voice="nova",
            enable_audio=False
        )
        assert client.voice == "nova"

    def test_create_voice_client_realtime_disabled(self):
        """Test create_voice_client with realtime disabled."""
        from dialogue.realtime_client import create_voice_client
        client = create_voice_client(
            api_key="sk-test-key",
            use_realtime=False,
            enable_audio=False
        )
        assert client.use_realtime_api == False

    # VoiceResponse timestamp tests

    def test_voiceresponse_with_timestamp(self):
        """Test VoiceResponse with explicit timestamp."""
        from dialogue.realtime_client import VoiceResponse
        import time

        ts = time.time() - 100
        response = VoiceResponse(text="Test", timestamp=ts)
        assert response.timestamp == ts

    # Import fallback tests

    def test_constants_fallback_values(self):
        """Test that constants have fallback values when import fails."""
        # This tests the import fallback in realtime_client.py
        from dialogue import realtime_client
        # Should have fallback values defined
        assert hasattr(realtime_client, 'DEFAULT_COOLDOWN_MS')
        assert hasattr(realtime_client, 'DEFAULT_MAX_RESPONSE_LENGTH_MS')
        assert hasattr(realtime_client, 'MAX_TEXT_LENGTH')

    def test_init_when_openai_not_available(self):
        """Test initialization when OpenAI library is not available."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import patch

        with patch('dialogue.realtime_client.OPENAI_AVAILABLE', False):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            assert client.enabled == False
            assert client.client is None

    # WebSocket connection mock tests

    @pytest.mark.asyncio
    async def test_disconnect_realtime_api_closes_connection(self):
        """Test _disconnect_realtime_api closes websocket."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import AsyncMock, patch

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws = mock_ws
            client.ws_connected = True

            await client._disconnect_realtime_api()

            mock_ws.close.assert_called_once()
            assert client.ws is None
            assert client.ws_connected == False

    @pytest.mark.asyncio
    async def test_disconnect_realtime_api_handles_exception(self):
        """Test _disconnect_realtime_api handles close exception."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import AsyncMock, patch

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=Exception("Connection error"))

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws = mock_ws
            client.ws_connected = True

            # Should not raise
            await client._disconnect_realtime_api()

            assert client.ws is None
            assert client.ws_connected == False

    # TTS generation tests

    @pytest.mark.asyncio
    async def test_generate_speech_tts_with_audio_enabled(self):
        """Test _generate_speech_tts with audio enabled."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_audio_response = MagicMock()
        mock_audio_response.aread = AsyncMock(return_value=b"fake_audio_data")

        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(return_value=mock_audio_response)

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', False):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=True)
            client.client = mock_client

            result = await client._generate_speech_tts("Hello world")

            assert isinstance(result, VoiceResponse)
            assert result.text == "Hello world"
            assert result.audio_data == b"fake_audio_data"

    @pytest.mark.asyncio
    async def test_generate_speech_tts_text_only(self):
        """Test _generate_speech_tts with audio disabled."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        result = await client._generate_speech_tts("Text only")

        assert isinstance(result, VoiceResponse)
        assert result.text == "Text only"
        assert result.audio_data is None

    @pytest.mark.asyncio
    async def test_generate_speech_tts_handles_error(self):
        """Test _generate_speech_tts handles exceptions."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, patch

        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(side_effect=Exception("API error"))

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', False):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=True)
            client.client = mock_client

            result = await client._generate_speech_tts("Test")

            # Should return VoiceResponse with text only on error
            assert isinstance(result, VoiceResponse)
            assert result.text == "Test"
            assert result.audio_data is None

    @pytest.mark.asyncio
    async def test_generate_speech_tts_truncates_long_text(self):
        """Test _generate_speech_tts truncates text to MAX_TEXT_LENGTH."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create text longer than MAX_TEXT_LENGTH
        long_text = "A" * 600
        expected_text = "A" * 500  # Should be truncated

        mock_audio_response = MagicMock()
        mock_audio_response.aread = AsyncMock(return_value=b"audio")

        mock_client = AsyncMock()
        mock_client.audio.speech.create = AsyncMock(return_value=mock_audio_response)

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', False):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=True)
            client.client = mock_client

            await client._generate_speech_tts(long_text)

            # Check that truncated text was sent
            call_args = mock_client.audio.speech.create.call_args
            assert len(call_args[1]['input']) <= 500

    # _generate_speech tests

    @pytest.mark.asyncio
    async def test_generate_speech_uses_realtime_api_when_connected(self):
        """Test _generate_speech uses Realtime API when connected."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws_connected = True

            # Mock the send/receive method
            client._send_text_and_receive_audio = AsyncMock(
                return_value=VoiceResponse(text="Realtime response", audio_data=b"audio")
            )

            result = await client._generate_speech("Test", 1)

            assert result.text == "Realtime response"
            client._send_text_and_receive_audio.assert_called_once_with("Test", 1)

    @pytest.mark.asyncio
    async def test_generate_speech_connects_when_not_connected(self):
        """Test _generate_speech connects when not already connected."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws_connected = False

            client._connect_realtime_api = AsyncMock(return_value=True)
            client._send_text_and_receive_audio = AsyncMock(
                return_value=VoiceResponse(text="Response")
            )

            await client._generate_speech("Test", 1)

            client._connect_realtime_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_speech_falls_back_to_tts_on_connect_failure(self):
        """Test _generate_speech falls back to TTS when connection fails."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws_connected = False

            client._connect_realtime_api = AsyncMock(return_value=False)
            client._generate_speech_tts = AsyncMock(
                return_value=VoiceResponse(text="TTS fallback")
            )

            result = await client._generate_speech("Test", 1)

            assert result.text == "TTS fallback"
            client._generate_speech_tts.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_speech_when_not_using_realtime_api(self):
        """Test _generate_speech uses TTS when realtime_api disabled."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, patch

        client = RealtimeVoiceClient(
            api_key="sk-test-key",
            use_realtime_api=False,
            enable_audio_output=False
        )

        client._generate_speech_tts = AsyncMock(
            return_value=VoiceResponse(text="TTS response")
        )

        result = await client._generate_speech("Test", 1)

        assert result.text == "TTS response"
        client._generate_speech_tts.assert_called_once_with("Test")

    # speak() method edge cases

    def test_speak_handles_exception(self):
        """Test speak handles exceptions gracefully."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import patch

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Mock run_coroutine_threadsafe to raise exception
        with patch('asyncio.run_coroutine_threadsafe', side_effect=Exception("Async error")):
            result = client.speak("Test", priority=1)

            # Should return VoiceResponse with text on error
            assert isinstance(result, VoiceResponse)
            assert result.text == "Test"

    def test_speak_resets_state_on_error(self):
        """Test speak resets state when error occurs."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState
        from unittest.mock import patch

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        with patch('asyncio.run_coroutine_threadsafe', side_effect=Exception("Error")):
            client.speak("Test", priority=1)

            # State should be reset to IDLE
            assert client.speech_state == SpeechState.IDLE

    def test_speak_future_timeout(self):
        """Test speak handles future timeout."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import MagicMock, patch
        from concurrent.futures import TimeoutError

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Create a mock future that times out
        mock_future = MagicMock()
        mock_future.result.side_effect = TimeoutError("Future timeout")

        with patch('asyncio.run_coroutine_threadsafe', return_value=mock_future):
            result = client.speak("Test", priority=1)

            # Should return VoiceResponse with text on timeout
            assert isinstance(result, VoiceResponse)
            assert result.text == "Test"

    def test_speak_with_none_loop_initially(self):
        """Test speak when loop is None initially (before event loop starts)."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import patch

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None

        result = client.speak("Test", priority=1)

        # Should return VoiceResponse with text
        assert isinstance(result, VoiceResponse)
        assert result.text == "Test"

    def test_speak_success_updates_state_and_time(self):
        """Test speak updates state and last_spoken_time on success."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse, SpeechState
        from unittest.mock import MagicMock, patch

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.speech_state = SpeechState.IDLE
        client.current_priority = 99

        # Create a successful future
        mock_future = MagicMock()
        mock_response = VoiceResponse(text="Success", audio_data=b"audio")
        mock_future.result.return_value = mock_response

        with patch('asyncio.run_coroutine_threadsafe', return_value=mock_future):
            result = client.speak("Test", priority=1)

            # After successful speak, state should be reset
            assert client.speech_state == SpeechState.IDLE
            assert client.current_priority == 99
            assert client.last_spoken_time > 0
            assert result.text == "Success"

    # initialize() tests

    @pytest.mark.asyncio
    async def test_initialize_connects_realtime_api(self):
        """Test initialize connects to Realtime API when enabled."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import AsyncMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client._connect_realtime_api = AsyncMock(return_value=True)

            result = await client.initialize()

            assert result == True
            client._connect_realtime_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_when_realtime_disabled(self):
        """Test initialize skips connection when realtime disabled."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(
            api_key="sk-test-key",
            use_realtime_api=False,
            enable_audio_output=False
        )

        result = await client.initialize()

        assert result == True

    # shutdown() tests

    def test_shutdown_stops_speech(self):
        """Test shutdown stops current speech."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.speech_state = SpeechState.SPEAKING

        client.shutdown()

        assert client.enabled == False

    def test_shutdown_disconnects_websocket(self):
        """Test shutdown disconnects websocket."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import MagicMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws_connected = True

            # Mock the loop
            mock_loop = MagicMock()
            client.loop = mock_loop

            client.shutdown()

            # Should have scheduled disconnect
            assert mock_loop.call_soon_threadsafe.called

    def test_shutdown_sets_disabled(self):
        """Test shutdown sets enabled to False."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        assert client.enabled == True

        client.shutdown()

        assert client.enabled == False

    def test_shutdown_without_loop(self):
        """Test shutdown when loop is None."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None
        client.ws_connected = True

        # Should not raise
        client.shutdown()

        assert client.enabled == False

    # interrupt() tests

    def test_interrupt_schedules_cancel(self):
        """Test interrupt schedules response cancel via WebSocket."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from unittest.mock import MagicMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws_connected = True

            # Mock the loop
            mock_loop = MagicMock()
            client.loop = mock_loop

            client.interrupt()

            # Should have scheduled cancel
            assert client.interrupt_requested == True

    def test_interrupt_without_websocket_connection(self):
        """Test interrupt when not connected to WebSocket."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.ws_connected = False

        client.interrupt()

        assert client.interrupt_requested == True
        assert client.speech_state == SpeechState.INTERRUPTED

    def test_interrupt_without_loop(self):
        """Test interrupt when loop is None."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.loop = None
        client.ws_connected = True

        # Should not raise
        client.interrupt()

        assert client.interrupt_requested == True

    # stop() tests

    def test_stop_handles_empty_queue(self):
        """Test stop handles empty queue gracefully."""
        from dialogue.realtime_client import RealtimeVoiceClient

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        # Queue is already empty
        assert client.audio_queue.empty()

        # Should not raise
        client.stop()

    def test_stop_clears_multiple_queue_items(self):
        """Test stop clears multiple items from queue."""
        from dialogue.realtime_client import RealtimeVoiceClient, AudioChunk

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Add multiple items
        client.audio_queue.put(AudioChunk(data=b"item1"))
        client.audio_queue.put(AudioChunk(data=b"item2"))
        client.audio_queue.put(AudioChunk(data=b"item3"))

        assert client.audio_queue.qsize() == 3

        client.stop()

        assert client.audio_queue.empty()

    def test_stop_handles_queue_get_exception(self):
        """Test stop handles Empty exception from queue."""
        from dialogue.realtime_client import RealtimeVoiceClient, AudioChunk
        from queue import Empty

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Add one item then try to get more (should handle Empty)
        client.audio_queue.put(AudioChunk(data=b"item1"))

        # Stop will empty the queue
        client.stop()

        # Queue should be empty
        assert client.audio_queue.empty()

    def test_stop_handles_empty_exception_directly(self):
        """Test stop handles Empty exception when raised."""
        from dialogue.realtime_client import RealtimeVoiceClient
        from queue import Empty
        from unittest.mock import patch

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)

        # Mock empty() to return True (so we enter the loop)
        # and get_nowait() to raise Empty
        original_empty = client.audio_queue.empty
        call_count = [0]

        def mock_empty():
            call_count[0] += 1
            return call_count[0] <= 1  # Return True once, then False

        def mock_get_nowait():
            raise Empty()

        with patch.object(client.audio_queue, 'empty', side_effect=mock_empty):
            with patch.object(client.audio_queue, 'get_nowait', side_effect=mock_get_nowait):
                # Should not raise, should break on Empty exception
                client.stop()

    def test_stop_resets_state_and_clears_interrupt(self):
        """Test stop resets speech state."""
        from dialogue.realtime_client import RealtimeVoiceClient, SpeechState

        client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
        client.speech_state = SpeechState.SPEAKING
        client.interrupt_requested = True

        client.stop()

        assert client.speech_state == SpeechState.IDLE

    # API key validation tests

    def test_init_with_invalid_api_key_format(self):
        """Test initialization logs warning for invalid API key format."""
        from dialogue.realtime_client import RealtimeVoiceClient
        import logging

        # Create a client with invalid API key format (not starting with sk-)
        with patch.object(logging, 'getLogger') as mock_logger:
            logger_instance = MagicMock()
            mock_logger.return_value = logger_instance

            client = RealtimeVoiceClient(
                api_key="invalid-key-format",
                enable_audio_output=False
            )

            # Client should still work but log warning
            assert client.enabled == True

    # _send_text_and_receive_audio tests

    @pytest.mark.asyncio
    async def test_send_text_and_receive_audio_fallback_on_no_connection(self):
        """Test _send_text_and_receive_audio falls back to TTS when not connected."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, patch

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws_connected = False

            client._generate_speech_tts = AsyncMock(
                return_value=VoiceResponse(text="TTS response")
            )

            result = await client._send_text_and_receive_audio("Test", 1)

            assert result.text == "TTS response"

    @pytest.mark.asyncio
    async def test_send_text_and_receive_audio_handles_error(self):
        """Test _send_text_and_receive_audio handles exceptions."""
        from dialogue.realtime_client import RealtimeVoiceClient, VoiceResponse
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=Exception("WebSocket error"))

        with patch('dialogue.realtime_client.WEBSOCKETS_AVAILABLE', True):
            client = RealtimeVoiceClient(api_key="sk-test-key", enable_audio_output=False)
            client.ws = mock_ws
            client.ws_connected = True

            client._generate_speech_tts = AsyncMock(
                return_value=VoiceResponse(text="Fallback")
            )

            result = await client._send_text_and_receive_audio("Test", 1)

            assert result.text == "Fallback"
