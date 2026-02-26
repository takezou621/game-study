"""Tests for domain events."""

import importlib.util
from pathlib import Path

# Direct module imports to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"

# Load base event
base_event_spec = importlib.util.spec_from_file_location(
    "base_event",
    SRC_PATH / "domain" / "events" / "base.py",
)
base_event_module = importlib.util.module_from_spec(base_event_spec)
base_event_spec.loader.exec_module(base_event_module)

# Load game events
game_events_spec = importlib.util.spec_from_file_location(
    "game_events",
    SRC_PATH / "domain" / "events" / "game_events.py",
)
game_events_module = importlib.util.module_from_spec(game_events_spec)
game_events_spec.loader.exec_module(game_events_module)

# Load audio events
audio_events_spec = importlib.util.spec_from_file_location(
    "audio_events",
    SRC_PATH / "domain" / "events" / "audio_events.py",
)
audio_events_module = importlib.util.module_from_spec(audio_events_spec)
audio_events_spec.loader.exec_module(audio_events_module)

# Import classes
DomainEvent = base_event_module.DomainEvent

GameStateChanged = game_events_module.GameStateChanged
TriggerFired = game_events_module.TriggerFired
PlayerStatusChanged = game_events_module.PlayerStatusChanged
StormStatusChanged = game_events_module.StormStatusChanged

SpeechDetected = audio_events_module.SpeechDetected
AudioChunkEvent = audio_events_module.AudioChunkEvent
SpeechInterrupted = audio_events_module.SpeechInterrupted


class TestDomainEvent:
    """Tests for base DomainEvent class."""

    def test_init_generates_event_id(self):
        """Test DomainEvent generates unique event_id."""
        event = DomainEvent()
        assert event.event_id is not None
        assert len(event.event_id) > 0

    def test_init_generates_timestamp(self):
        """Test DomainEvent generates timestamp."""
        event = DomainEvent()
        # Just verify it has a timestamp (actual value will vary)
        assert event.timestamp_ms is not None
        assert event.timestamp_ms > 0

    def test_default_event_type(self):
        """Test default event_type."""
        event = DomainEvent()
        assert event.event_type == "domain_event"

    def test_unique_event_ids(self):
        """Test each event gets unique ID."""
        event1 = DomainEvent()
        event2 = DomainEvent()
        assert event1.event_id != event2.event_id

    def test_to_dict(self):
        """Test to_dict method."""
        event = DomainEvent()
        result = event.to_dict()
        assert "event_id" in result
        assert "timestamp_ms" in result
        assert "event_type" in result
        assert result["event_type"] == "domain_event"


class TestGameStateChanged:
    """Tests for GameStateChanged event."""

    def test_init_defaults(self):
        """Test GameStateChanged initialization with defaults."""
        event = GameStateChanged()
        assert event.event_type == "game_state_changed"
        assert event.previous_state == {}
        assert event.current_state == {}
        assert event.changed_fields == []

    def test_init_with_values(self):
        """Test GameStateChanged initialization with values."""
        event = GameStateChanged(
            previous_state={"hp": 100},
            current_state={"hp": 75},
            changed_fields=["hp"],
        )
        assert event.previous_state == {"hp": 100}
        assert event.current_state == {"hp": 75}
        assert event.changed_fields == ["hp"]

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = GameStateChanged(
            previous_state={"hp": 100},
            current_state={"hp": 50},
            changed_fields=["hp", "shield"],
        )
        result = event.to_dict()
        assert result["event_type"] == "game_state_changed"
        assert result["previous_state"] == {"hp": 100}
        assert result["current_state"] == {"hp": 50}
        assert "hp" in result["changed_fields"]
        assert "shield" in result["changed_fields"]
        assert "event_id" in result
        assert "timestamp_ms" in result


class TestTriggerFired:
    """Tests for TriggerFired event."""

    def test_init_defaults(self):
        """Test TriggerFired initialization with defaults."""
        event = TriggerFired()
        assert event.event_type == "trigger_fired"
        assert event.trigger_id == ""
        assert event.trigger_name == ""
        assert event.priority == 0
        assert event.template is None
        assert event.movement_state == "non_combat"
        assert event.game_state_snapshot == {}

    def test_init_with_values(self):
        """Test TriggerFired initialization with values."""
        event = TriggerFired(
            trigger_id="low_hp",
            trigger_name="Low HP Alert",
            priority=2,
            template="Heal now!",
            movement_state="combat",
            game_state_snapshot={"hp": 25},
        )
        assert event.trigger_id == "low_hp"
        assert event.trigger_name == "Low HP Alert"
        assert event.priority == 2
        assert event.template == "Heal now!"
        assert event.movement_state == "combat"
        assert event.game_state_snapshot == {"hp": 25}

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = TriggerFired(
            trigger_id="in_storm",
            trigger_name="In Storm Warning",
            priority=3,
            template="Get out of storm!",
            movement_state="combat",
        )
        result = event.to_dict()
        assert result["event_type"] == "trigger_fired"
        assert result["trigger_id"] == "in_storm"
        assert result["trigger_name"] == "In Storm Warning"
        assert result["priority"] == 3
        assert result["template"] == "Get out of storm!"
        assert result["movement_state"] == "combat"


class TestPlayerStatusChanged:
    """Tests for PlayerStatusChanged event."""

    def test_init_defaults(self):
        """Test PlayerStatusChanged initialization with defaults."""
        event = PlayerStatusChanged()
        assert event.event_type == "player_status_changed"
        assert event.hp_previous is None
        assert event.hp_current is None
        assert event.shield_previous is None
        assert event.shield_current is None
        assert event.is_knocked_previous is False
        assert event.is_knocked_current is False

    def test_init_with_values(self):
        """Test PlayerStatusChanged initialization with values."""
        event = PlayerStatusChanged(
            hp_previous=100,
            hp_current=50,
            shield_previous=50,
            shield_current=0,
            is_knocked_previous=False,
            is_knocked_current=True,
        )
        assert event.hp_previous == 100
        assert event.hp_current == 50
        assert event.shield_previous == 50
        assert event.shield_current == 0
        assert event.is_knocked_previous is False
        assert event.is_knocked_current is True

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = PlayerStatusChanged(
            hp_previous=75,
            hp_current=50,
            is_knocked_current=True,
        )
        result = event.to_dict()
        assert result["event_type"] == "player_status_changed"
        assert result["hp_previous"] == 75
        assert result["hp_current"] == 50
        assert result["is_knocked_previous"] is False
        assert result["is_knocked_current"] is True


class TestStormStatusChanged:
    """Tests for StormStatusChanged event."""

    def test_init_defaults(self):
        """Test StormStatusChanged initialization with defaults."""
        event = StormStatusChanged()
        assert event.event_type == "storm_status_changed"
        assert event.in_storm_previous is False
        assert event.in_storm_current is False
        assert event.is_shrinking_previous is False
        assert event.is_shrinking_current is False
        assert event.phase_previous is None
        assert event.phase_current is None

    def test_init_with_values(self):
        """Test StormStatusChanged initialization with values."""
        event = StormStatusChanged(
            in_storm_previous=False,
            in_storm_current=True,
            is_shrinking_previous=False,
            is_shrinking_current=True,
            phase_previous=2,
            phase_current=3,
        )
        assert event.in_storm_previous is False
        assert event.in_storm_current is True
        assert event.is_shrinking_previous is False
        assert event.is_shrinking_current is True
        assert event.phase_previous == 2
        assert event.phase_current == 3

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = StormStatusChanged(
            in_storm_current=True,
            phase_current=4,
        )
        result = event.to_dict()
        assert result["event_type"] == "storm_status_changed"
        assert result["in_storm_current"] is True
        assert result["phase_current"] == 4


class TestSpeechDetected:
    """Tests for SpeechDetected event."""

    def test_init_defaults(self):
        """Test SpeechDetected initialization with defaults."""
        event = SpeechDetected()
        assert event.event_type == "speech_detected"
        assert event.duration_ms == 0
        assert event.confidence == 0.0
        assert event.audio_data_size == 0

    def test_init_with_values(self):
        """Test SpeechDetected initialization with values."""
        event = SpeechDetected(
            duration_ms=1500,
            confidence=0.95,
            audio_data_size=48000,
        )
        assert event.duration_ms == 1500
        assert event.confidence == 0.95
        assert event.audio_data_size == 48000

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = SpeechDetected(
            duration_ms=2000,
            confidence=0.8,
        )
        result = event.to_dict()
        assert result["event_type"] == "speech_detected"
        assert result["duration_ms"] == 2000
        assert result["confidence"] == 0.8


class TestAudioChunkEvent:
    """Tests for AudioChunkEvent event."""

    def test_init_defaults(self):
        """Test AudioChunkEvent initialization with defaults."""
        event = AudioChunkEvent()
        assert event.event_type == "audio_chunk"
        assert event.chunk_size == 0
        assert event.sample_rate == 16000
        assert event.channels == 1
        assert event.is_speech is False

    def test_init_with_values(self):
        """Test AudioChunkEvent initialization with values."""
        event = AudioChunkEvent(
            chunk_size=1024,
            sample_rate=48000,
            channels=2,
            is_speech=True,
        )
        assert event.chunk_size == 1024
        assert event.sample_rate == 48000
        assert event.channels == 2
        assert event.is_speech is True

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = AudioChunkEvent(
            chunk_size=512,
            is_speech=True,
        )
        result = event.to_dict()
        assert result["event_type"] == "audio_chunk"
        assert result["chunk_size"] == 512
        assert result["sample_rate"] == 16000
        assert result["channels"] == 1
        assert result["is_speech"] is True


class TestSpeechInterrupted:
    """Tests for SpeechInterrupted event."""

    def test_init_defaults(self):
        """Test SpeechInterrupted initialization with defaults."""
        event = SpeechInterrupted()
        assert event.event_type == "speech_interrupted"
        assert event.original_duration_ms == 0
        assert event.interrupted_at_ms == 0
        assert event.remaining_ms == 0
        assert event.priority == 0

    def test_init_with_values(self):
        """Test SpeechInterrupted initialization with values."""
        event = SpeechInterrupted(
            original_duration_ms=5000,
            interrupted_at_ms=2000,
            remaining_ms=3000,
            priority=3,
        )
        assert event.original_duration_ms == 5000
        assert event.interrupted_at_ms == 2000
        assert event.remaining_ms == 3000
        assert event.priority == 3

    def test_to_dict(self):
        """Test to_dict includes all fields."""
        event = SpeechInterrupted(
            original_duration_ms=3000,
            remaining_ms=1500,
        )
        result = event.to_dict()
        assert result["event_type"] == "speech_interrupted"
        assert result["original_duration_ms"] == 3000
        assert result["remaining_ms"] == 1500


class TestEventInheritance:
    """Tests for event class inheritance."""

    def test_game_state_changed_has_event_type(self):
        """Test GameStateChanged has correct event_type."""
        event = GameStateChanged()
        assert event.event_type == "game_state_changed"

    def test_trigger_fired_has_event_type(self):
        """Test TriggerFired has correct event_type."""
        event = TriggerFired()
        assert event.event_type == "trigger_fired"

    def test_player_status_changed_has_event_type(self):
        """Test PlayerStatusChanged has correct event_type."""
        event = PlayerStatusChanged()
        assert event.event_type == "player_status_changed"

    def test_storm_status_changed_has_event_type(self):
        """Test StormStatusChanged has correct event_type."""
        event = StormStatusChanged()
        assert event.event_type == "storm_status_changed"

    def test_speech_detected_has_event_type(self):
        """Test SpeechDetected has correct event_type."""
        event = SpeechDetected()
        assert event.event_type == "speech_detected"

    def test_audio_chunk_event_has_event_type(self):
        """Test AudioChunkEvent has correct event_type."""
        event = AudioChunkEvent()
        assert event.event_type == "audio_chunk"

    def test_speech_interrupted_has_event_type(self):
        """Test SpeechInterrupted has correct event_type."""
        event = SpeechInterrupted()
        assert event.event_type == "speech_interrupted"

    def test_to_dict_includes_base_fields(self):
        """Test to_dict includes base event fields."""
        event = TriggerFired(trigger_id="test")
        result = event.to_dict()
        # Base fields
        assert "event_id" in result
        assert "timestamp_ms" in result
        assert "event_type" in result
        # Derived fields
        assert "trigger_id" in result
