"""Tests for voice types module."""

import time

import pytest

from dialogue.voice_types import (
    COMBAT_TEMPLATES,
    AudioChunk,
    SpeechState,
    VoiceResponse,
)


class TestSpeechState:
    """Tests for SpeechState enum."""

    def test_has_idle_state(self) -> None:
        """Test that IDLE state exists."""
        assert SpeechState.IDLE.value == "idle"

    def test_has_speaking_state(self) -> None:
        """Test that SPEAKING state exists."""
        assert SpeechState.SPEAKING.value == "speaking"

    def test_has_interrupted_state(self) -> None:
        """Test that INTERRUPTED state exists."""
        assert SpeechState.INTERRUPTED.value == "interrupted"

    def test_all_states_are_unique(self) -> None:
        """Test that all states have unique values."""
        values = [state.value for state in SpeechState]
        assert len(values) == len(set(values))


class TestVoiceResponse:
    """Tests for VoiceResponse dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic VoiceResponse."""
        response = VoiceResponse(text="Hello")
        assert response.text == "Hello"
        assert response.audio_data is None
        assert response.duration_ms is None
        assert response.priority == 2
        assert response.interrupted is False

    def test_auto_timestamp(self) -> None:
        """Test that timestamp is auto-generated."""
        before = time.time()
        response = VoiceResponse(text="Test")
        after = time.time()

        assert response.timestamp is not None
        assert before <= response.timestamp <= after

    def test_custom_timestamp(self) -> None:
        """Test setting a custom timestamp."""
        custom_ts = 12345.0
        response = VoiceResponse(text="Test", timestamp=custom_ts)
        assert response.timestamp == custom_ts

    def test_with_audio_data(self) -> None:
        """Test VoiceResponse with audio data."""
        audio = b"fake_audio_data"
        response = VoiceResponse(text="Hello", audio_data=audio, duration_ms=1000)
        assert response.audio_data == audio
        assert response.duration_ms == 1000

    def test_priority_levels(self) -> None:
        """Test different priority levels."""
        for priority in [0, 1, 2, 3]:
            response = VoiceResponse(text="Test", priority=priority)
            assert response.priority == priority

    def test_interrupted_flag(self) -> None:
        """Test interrupted flag."""
        response = VoiceResponse(text="Test", interrupted=True)
        assert response.interrupted is True


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic AudioChunk."""
        chunk = AudioChunk(data=b"audio_data")
        assert chunk.data == b"audio_data"
        assert chunk.timestamp is not None

    def test_auto_timestamp(self) -> None:
        """Test that timestamp is auto-generated."""
        before = time.time()
        chunk = AudioChunk(data=b"data")
        after = time.time()

        assert before <= chunk.timestamp <= after

    def test_custom_timestamp(self) -> None:
        """Test setting a custom timestamp."""
        custom_ts = 12345.0
        chunk = AudioChunk(data=b"data", timestamp=custom_ts)
        assert chunk.timestamp == custom_ts


class TestCombatTemplates:
    """Tests for COMBAT_TEMPLATES constant."""

    def test_p0_templates_exist(self) -> None:
        """Test that P0 (Survival) templates exist."""
        assert 0 in COMBAT_TEMPLATES
        assert "low_hp" in COMBAT_TEMPLATES[0]
        assert "knocked" in COMBAT_TEMPLATES[0]
        assert "storm" in COMBAT_TEMPLATES[0]

    def test_p1_templates_exist(self) -> None:
        """Test that P1 (Tactical) templates exist."""
        assert 1 in COMBAT_TEMPLATES
        assert "rotate" in COMBAT_TEMPLATES[1]
        assert "storm_shrinking" in COMBAT_TEMPLATES[1]

    def test_templates_are_short(self) -> None:
        """Test that all templates are short (for combat situations)."""
        for priority, templates in COMBAT_TEMPLATES.items():
            for key, template in templates.items():
                # Combat templates should be concise
                assert len(template) <= 30, f"Template too long: {template}"
                # Should end with exclamation for urgency
                assert template.endswith("!"), f"Template should end with '!': {template}"

    def test_no_p2_or_higher_templates(self) -> None:
        """Test that there are no P2 or higher templates."""
        for priority in COMBAT_TEMPLATES:
            assert priority in [0, 1], f"Unexpected priority level: {priority}"
