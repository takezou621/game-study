#!/usr/bin/env python3
"""Unit tests for RealtimeVoiceClient."""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dialogue.realtime_client import (
    AudioChunk,
    RealtimeVoiceClient,
    SpeechState,
    VoiceResponse,
    create_voice_client,
)


class TestVoiceResponse(unittest.TestCase):
    """Test VoiceResponse dataclass."""

    def test_create_voice_response(self):
        """Test creating a VoiceResponse."""
        response = VoiceResponse(text="Hello")
        self.assertEqual(response.text, "Hello")
        self.assertIsNone(response.audio_data)
        self.assertIsNone(response.duration_ms)
        self.assertIsNotNone(response.timestamp)

    def test_voice_response_with_audio(self):
        """Test VoiceResponse with audio data."""
        audio = b"fake_audio_data"
        response = VoiceResponse(text="Hello", audio_data=audio, duration_ms=1000)
        self.assertEqual(response.audio_data, audio)
        self.assertEqual(response.duration_ms, 1000)


class TestAudioChunk(unittest.TestCase):
    """Test AudioChunk dataclass."""

    def test_create_audio_chunk(self):
        """Test creating an AudioChunk."""
        chunk = AudioChunk(data=b"audio_data")
        self.assertEqual(chunk.data, b"audio_data")
        self.assertIsNotNone(chunk.timestamp)


class TestRealtimeVoiceClient(unittest.TestCase):
    """Test RealtimeVoiceClient class."""

    def test_create_client_without_api_key(self):
        """Test creating client without API key."""
        with patch.dict(os.environ, {}, clear=True):
            client = RealtimeVoiceClient(api_key=None)
            # Client should be disabled without API key
            # But it won't be fully initialized without the event loop
            self.assertIsNotNone(client)

    def test_create_client_with_api_key(self):
        """Test creating client with API key."""
        with patch("dialogue.realtime_client.OPENAI_AVAILABLE", True):
            client = RealtimeVoiceClient(api_key="test_key", enable_audio_output=False)
            self.assertEqual(client.api_key, "test_key")
            self.assertEqual(client.cooldown_ms, 3000)
            self.assertEqual(client.max_response_length_ms, 10000)
            client.shutdown()

    def test_load_system_prompt(self):
        """Test loading system prompt."""
        client = RealtimeVoiceClient(api_key="test_key", enable_audio_output=False)
        prompt = client.system_prompt
        self.assertIn("English teacher", prompt)
        self.assertIn("Priority Levels", prompt)
        client.shutdown()

    def test_get_short_response_combat(self):
        """Test getting short response during combat."""
        client = RealtimeVoiceClient(api_key="test_key", enable_audio_output=False)

        # Test P0 low_hp in combat
        trigger_info = {"priority": 0, "rule_id": "p0_low_hp"}
        short = client._get_short_response(trigger_info, "combat")
        self.assertEqual(short, "Low HP! Cover!")

        # Test P0 knocked in combat
        trigger_info = {"priority": 0, "rule_id": "p0_knocked"}
        short = client._get_short_response(trigger_info, "combat")
        self.assertEqual(short, "Knocked! Ping!")

        # Test non-combat should return None
        short = client._get_short_response(trigger_info, "non_combat")
        self.assertIsNone(short)

        client.shutdown()

    def test_enhance_template_combat(self):
        """Test template enhancement during combat."""
        client = RealtimeVoiceClient(api_key="test_key", enable_audio_output=False)

        # Long template should be truncated in combat
        long_template = "Your health is very low. You should find cover immediately and heal up."
        enhanced = client._enhance_template(long_template, {}, "combat")
        self.assertLessEqual(len(enhanced), 55)  # First sentence + !

        # Short template should remain unchanged
        short_template = "Low HP!"
        enhanced = client._enhance_template(short_template, {}, "combat")
        self.assertEqual(enhanced, "Low HP!")

        # Non-combat should not truncate
        enhanced = client._enhance_template(long_template, {}, "non_combat")
        self.assertEqual(enhanced, long_template)

        client.shutdown()

    def test_speak_disabled_client(self):
        """Test speak with disabled client."""
        with patch.dict(os.environ, {}, clear=True):
            client = RealtimeVoiceClient(api_key=None)
            response = client.speak("Hello")
            self.assertEqual(response.text, "Hello")
            self.assertIsNone(response.audio_data)

    def test_speak_with_low_priority_during_cooldown(self):
        """Test that low priority speech is skipped during cooldown."""
        client = RealtimeVoiceClient(
            api_key="test_key", enable_audio_output=False, cooldown_ms=5000
        )
        client.last_spoken_time = time.time()
        client.enabled = True

        # P2 should be skipped during cooldown
        response = client.speak("Hello", priority=2)
        self.assertIsNone(response)

        client.shutdown()

    def test_interrupt(self):
        """Test interrupt functionality."""
        client = RealtimeVoiceClient(api_key="test_key", enable_audio_output=False)

        client.speech_state = SpeechState.SPEAKING
        client.interrupt()

        self.assertTrue(client.interrupt_requested)
        self.assertEqual(client.speech_state, SpeechState.INTERRUPTED)

        client.shutdown()

    def test_stop(self):
        """Test stop functionality."""
        client = RealtimeVoiceClient(api_key="test_key", enable_audio_output=False)

        # Add some audio to queue
        client.audio_queue.put(AudioChunk(data=b"test"))
        client.speech_state = SpeechState.SPEAKING

        client.stop()

        self.assertEqual(client.speech_state, SpeechState.IDLE)
        self.assertTrue(client.audio_queue.empty())

        client.shutdown()


class TestCreateVoiceClient(unittest.TestCase):
    """Test create_voice_client factory function."""

    def test_create_voice_client_defaults(self):
        """Test creating client with defaults."""
        client = create_voice_client(api_key="test_key")

        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.voice, "alloy")
        self.assertTrue(client.enable_audio_output)

        client.shutdown()

    def test_create_voice_client_custom(self):
        """Test creating client with custom options."""
        client = create_voice_client(
            api_key="test_key", voice="nova", enable_audio=False, use_realtime=False
        )

        self.assertEqual(client.voice, "nova")
        self.assertFalse(client.enable_audio_output)
        self.assertFalse(client.use_realtime_api)

        client.shutdown()


class TestSpeechState(unittest.TestCase):
    """Test SpeechState enum."""

    def test_speech_states(self):
        """Test all speech states exist."""
        self.assertEqual(SpeechState.IDLE.value, "idle")
        self.assertEqual(SpeechState.SPEAKING.value, "speaking")
        self.assertEqual(SpeechState.INTERRUPTED.value, "interrupted")


if __name__ == "__main__":
    unittest.main()
