"""Tests for application services: GameAnalyzerService, VoiceCoachService."""

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Direct module imports to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"

# Load game analyzer
game_analyzer_spec = importlib.util.spec_from_file_location(
    "game_analyzer",
    SRC_PATH / "application" / "services" / "game_analyzer.py",
)
game_analyzer_module = importlib.util.module_from_spec(game_analyzer_spec)
game_analyzer_spec.loader.exec_module(game_analyzer_module)

# Load voice coach
voice_coach_spec = importlib.util.spec_from_file_location(
    "voice_coach",
    SRC_PATH / "application" / "services" / "voice_coach.py",
)
voice_coach_module = importlib.util.module_from_spec(voice_coach_spec)
voice_coach_spec.loader.exec_module(voice_coach_module)

# Load dependencies
game_state_spec = importlib.util.spec_from_file_location(
    "game_state",
    SRC_PATH / "domain" / "entities" / "game_state.py",
)
game_state_module = importlib.util.module_from_spec(game_state_spec)
game_state_spec.loader.exec_module(game_state_module)

player_spec = importlib.util.spec_from_file_location(
    "player",
    SRC_PATH / "domain" / "entities" / "player.py",
)
player_module = importlib.util.module_from_spec(player_spec)
player_spec.loader.exec_module(player_module)

session_spec = importlib.util.spec_from_file_location(
    "session",
    SRC_PATH / "domain" / "entities" / "session.py",
)
session_module = importlib.util.module_from_spec(session_spec)
session_spec.loader.exec_module(session_module)

health_spec = importlib.util.spec_from_file_location(
    "health",
    SRC_PATH / "domain" / "value_objects" / "health.py",
)
health_module = importlib.util.module_from_spec(health_spec)
health_spec.loader.exec_module(health_module)

ammo_spec = importlib.util.spec_from_file_location(
    "ammo",
    SRC_PATH / "domain" / "value_objects" / "ammo.py",
)
ammo_module = importlib.util.module_from_spec(ammo_spec)
ammo_spec.loader.exec_module(ammo_module)

state_validator_spec = importlib.util.spec_from_file_location(
    "state_validator",
    SRC_PATH / "domain" / "services" / "state_validator.py",
)
state_validator_module = importlib.util.module_from_spec(state_validator_spec)
state_validator_spec.loader.exec_module(state_validator_module)

# Import classes
GameAnalyzerService = game_analyzer_module.GameAnalyzerService
GameAnalysis = game_analyzer_module.GameAnalysis

VoiceCoachService = voice_coach_module.VoiceCoachService
CoachState = voice_coach_module.CoachState

GameState = game_state_module.GameState
Player = game_state_module.Player
WeaponInfo = game_state_module.WeaponInfo
InventoryInfo = game_state_module.InventoryInfo
StormInfo = game_state_module.StormInfo
WorldInfo = game_state_module.WorldInfo

PlayerStatus = player_module.PlayerStatus
Session = session_module.Session
SessionPhase = session_module.SessionPhase

HP = health_module.HP
Shield = health_module.Shield
Ammo = ammo_module.Ammo

StateValidator = state_validator_module.StateValidator


class TestGameAnalysis:
    """Tests for GameAnalysis dataclass."""

    def test_init_defaults(self):
        """Test GameAnalysis initialization with defaults."""
        analysis = GameAnalysis(
            urgency_level=0,
            is_combat=False,
            needs_attention=False,
        )
        assert analysis.urgency_level == 0
        assert analysis.is_combat is False
        assert analysis.needs_attention is False
        assert analysis.recommendations == []
        assert analysis.key_metrics == {}

    def test_init_with_values(self):
        """Test GameAnalysis initialization with values."""
        analysis = GameAnalysis(
            urgency_level=3,
            is_combat=True,
            needs_attention=True,
            recommendations=["Heal now!", "Take cover!"],
            key_metrics={"hp": 20, "shield": 0},
        )
        assert analysis.urgency_level == 3
        assert analysis.is_combat is True
        assert analysis.needs_attention is True
        assert len(analysis.recommendations) == 2
        assert analysis.key_metrics["hp"] == 20


class TestGameAnalyzerService:
    """Tests for GameAnalyzerService."""

    @pytest.fixture
    def healthy_state(self) -> GameState:
        """Create a healthy game state."""
        return GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=100),
                    shield=Shield(value=100),
                    is_knocked=False,
                ),
                weapon=WeaponInfo(name="AR", ammo=Ammo(value=30)),
                inventory=InventoryInfo(materials=500),
            ),
            world=WorldInfo(storm=StormInfo(in_storm=False)),
            session=Session(phase=SessionPhase.MID_GAME),
        )

    @pytest.fixture
    def critical_state(self) -> GameState:
        """Create a critical game state."""
        return GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=20),
                    shield=Shield(value=0),
                    is_knocked=False,
                ),
                weapon=WeaponInfo(name="AR", ammo=Ammo(value=5, max_value=30)),
                inventory=InventoryInfo(materials=50),
            ),
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=3.0)),
            session=Session(phase=SessionPhase.LATE_GAME),
        )

    @pytest.fixture
    def service(self) -> GameAnalyzerService:
        """Create a game analyzer service."""
        return GameAnalyzerService()

    def test_init(self, service: GameAnalyzerService):
        """Test service initialization."""
        assert service.state_validator is not None

    def test_analyze_healthy_state(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test analyze with healthy state."""
        analysis = service.analyze(healthy_state)
        assert analysis.urgency_level == 0
        assert analysis.is_combat is False
        assert analysis.needs_attention is False

    def test_analyze_critical_state(self, service: GameAnalyzerService, critical_state: GameState):
        """Test analyze with critical state."""
        analysis = service.analyze(critical_state)
        assert analysis.urgency_level >= 2
        assert analysis.is_combat is True
        assert analysis.needs_attention is True

    def test_analyze_recommendations_critical_hp(self, service: GameAnalyzerService):
        """Test analyze generates critical HP recommendation."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=15))),
        )
        analysis = service.analyze(state)
        assert any("critical hp" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_low_hp(self, service: GameAnalyzerService):
        """Test analyze generates low HP recommendation."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=40))),
        )
        analysis = service.analyze(state)
        assert any("heal" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_in_storm(self, service: GameAnalyzerService):
        """Test analyze generates storm recommendation."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=2.0)),
        )
        analysis = service.analyze(state)
        assert any("storm" in r.lower() or "safe zone" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_heavy_storm(self, service: GameAnalyzerService):
        """Test analyze generates heavy storm recommendation."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=5.0)),
        )
        analysis = service.analyze(state)
        assert any("heavy damage" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_storm_shrinking(self, service: GameAnalyzerService):
        """Test analyze generates shrinking storm recommendation."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(is_shrinking=True)),
        )
        analysis = service.analyze(state)
        assert any("shrinking" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_knocked(self, service: GameAnalyzerService):
        """Test analyze generates knocked recommendation."""
        state = GameState(
            player=Player(status=PlayerStatus(is_knocked=True)),
        )
        analysis = service.analyze(state)
        assert any("revive" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_low_ammo(self, service: GameAnalyzerService):
        """Test analyze generates low ammo recommendation."""
        state = GameState(
            player=Player(
                weapon=WeaponInfo(name="AR", ammo=Ammo(value=5, max_value=30)),
            ),
        )
        analysis = service.analyze(state)
        assert any("ammo" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_no_ammo(self, service: GameAnalyzerService):
        """Test analyze generates out of ammo recommendation."""
        state = GameState(
            player=Player(
                weapon=WeaponInfo(name="AR", ammo=Ammo(value=0)),
            ),
        )
        analysis = service.analyze(state)
        assert any("out of ammo" in r.lower() or "switch" in r.lower() for r in analysis.recommendations)

    def test_analyze_recommendations_low_materials(self, service: GameAnalyzerService):
        """Test analyze generates farm materials recommendation."""
        state = GameState(
            player=Player(inventory=InventoryInfo(materials=50)),
        )
        analysis = service.analyze(state)
        assert any("farm" in r.lower() or "materials" in r.lower() for r in analysis.recommendations)

    def test_analyze_key_metrics(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test analyze extracts key metrics."""
        analysis = service.analyze(healthy_state)
        assert analysis.key_metrics["hp"] == 100
        assert analysis.key_metrics["shield"] == 100
        assert analysis.key_metrics["total_health"] == 200
        assert analysis.key_metrics["is_knocked"] is False
        assert analysis.key_metrics["in_storm"] is False
        assert analysis.key_metrics["weapon"] == "AR"
        assert analysis.key_metrics["ammo"] == 30
        assert analysis.key_metrics["materials"] == 500
        assert analysis.key_metrics["session_phase"] == "mid_game"

    def test_get_movement_state_combat(self, service: GameAnalyzerService):
        """Test get_movement_state returns combat for critical state."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=30))),
        )
        assert service.get_movement_state(state) == "combat"

    def test_get_movement_state_non_combat(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test get_movement_state returns non_combat for healthy state."""
        assert service.get_movement_state(healthy_state) == "non_combat"

    def test_compare_states_hp_change(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test compare_states detects HP change."""
        current = healthy_state.with_hp(75)
        changes = service.compare_states(healthy_state, current)
        assert "hp" in changes
        assert changes["hp"]["old"] == 100
        assert changes["hp"]["new"] == 75
        assert changes["hp"]["diff"] == -25
        assert changes["hp"]["is_damage"] is True

    def test_compare_states_shield_change(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test compare_states detects shield change."""
        current = healthy_state.with_shield(50)
        changes = service.compare_states(healthy_state, current)
        assert "shield" in changes
        assert changes["shield"]["diff"] == -50

    def test_compare_states_knocked_change(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test compare_states detects knocked change."""
        current = healthy_state.with_knocked(True)
        changes = service.compare_states(healthy_state, current)
        assert "knocked" in changes
        assert changes["knocked"]["old"] is False
        assert changes["knocked"]["new"] is True

    def test_compare_states_storm_change(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test compare_states detects storm change."""
        current = healthy_state.with_storm(in_storm=True)
        changes = service.compare_states(healthy_state, current)
        assert "in_storm" in changes
        assert changes["in_storm"]["old"] is False
        assert changes["in_storm"]["new"] is True

    def test_compare_states_no_change(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test compare_states returns empty dict when no change."""
        changes = service.compare_states(healthy_state, healthy_state)
        assert changes == {}

    def test_should_suppress_response_recent_response(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test should_suppress_response returns True for recent response."""
        # Get current time and use it as last_response to simulate very recent response
        # With elapsed time = 0 and min_interval > 0, should suppress
        from utils.time import get_timestamp_ms
        now = get_timestamp_ms()
        result = service.should_suppress_response(
            healthy_state,
            last_response_ms=now,  # Just now
            min_interval_ms=10000,  # 10 seconds
        )
        assert result is True

    def test_should_suppress_response_enough_time(self, service: GameAnalyzerService, healthy_state: GameState):
        """Test should_suppress_response returns False after enough time."""
        # Use 0 as last_response_ms (very old), and small min_interval
        result = service.should_suppress_response(
            healthy_state,
            last_response_ms=0,  # Very old timestamp
            min_interval_ms=1,  # Very small interval
        )
        assert result is False

    def test_should_suppress_response_urgent_bypasses(self, service: GameAnalyzerService):
        """Test should_suppress_response allows urgent responses."""
        critical_state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=20))),
        )
        # Urgent state should not suppress if enough time has passed (>=1000ms)
        # Using 0 as last_response_ms means a lot of time has passed
        result = service.should_suppress_response(
            critical_state,
            last_response_ms=0,  # Very old timestamp
            min_interval_ms=999999999,  # Large normal interval
        )
        # Urgent states bypass the normal interval if elapsed >= 1000ms
        assert result is False


class TestCoachState:
    """Tests for CoachState enum."""

    def test_coach_state_values(self):
        """Test CoachState enum values."""
        assert CoachState.IDLE.value == "idle"
        assert CoachState.SPEAKING.value == "speaking"
        assert CoachState.PROCESSING.value == "processing"


class TestVoiceCoachService:
    """Tests for VoiceCoachService."""

    @pytest.fixture
    def mock_evaluate_usecase(self):
        """Create mock evaluate usecase."""
        mock = MagicMock()
        mock.execute.return_value = MagicMock(
            has_firing_triggers=False,
            selected_trigger=None,
        )
        mock.evaluator = MagicMock()
        mock.evaluator.get_rule.return_value = None
        return mock

    @pytest.fixture
    def mock_generate_usecase(self):
        """Create mock generate usecase."""
        mock = MagicMock()
        mock.execute_from_trigger.return_value = None
        mock.execute_text_to_speech.return_value = None
        return mock

    @pytest.fixture
    def service(self, mock_evaluate_usecase, mock_generate_usecase) -> VoiceCoachService:
        """Create a voice coach service."""
        return VoiceCoachService(
            evaluate_usecase=mock_evaluate_usecase,
            generate_usecase=mock_generate_usecase,
        )

    @pytest.fixture
    def healthy_state(self) -> GameState:
        """Create a healthy game state."""
        return GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=100),
                    shield=Shield(value=100),
                )
            )
        )

    def test_init(self, service: VoiceCoachService):
        """Test service initialization."""
        assert service.state == CoachState.IDLE
        assert service.current_response is None
        assert service.response_queue == []
        assert service.max_queue_size == 3
        assert service.min_response_interval_ms == 2000

    def test_get_status_idle(self, service: VoiceCoachService):
        """Test get_status when idle."""
        status = service.get_status()
        assert status["state"] == "idle"
        assert status["current_response"] is None
        assert status["queue_size"] == 0

    def test_clear_queue_empty(self, service: VoiceCoachService):
        """Test clear_queue when empty."""
        count = service.clear_queue()
        assert count == 0

    def test_clear_queue_with_items(self, service: VoiceCoachService):
        """Test clear_queue with items."""
        # Add mock items to queue
        from application.dto.response_dto import QueuedResponseDTO, AudioResponseDTO

        mock_response = AudioResponseDTO(text="test", priority=0)
        service.response_queue = [
            QueuedResponseDTO(response=mock_response, queue_position=0),
            QueuedResponseDTO(response=mock_response, queue_position=1),
        ]
        count = service.clear_queue()
        assert count == 2
        assert service.response_queue == []

    def test_shutdown(self, service: VoiceCoachService):
        """Test shutdown clears state."""
        service.state = CoachState.SPEAKING
        service.shutdown()
        assert service.state == CoachState.IDLE
        assert service.current_response is None
        assert service.response_queue == []

    def test_process_state_no_firing_triggers(
        self, service: VoiceCoachService, healthy_state: GameState, mock_evaluate_usecase
    ):
        """Test process_state returns None when no triggers fire."""
        mock_evaluate_usecase.execute.return_value = MagicMock(
            has_firing_triggers=False,
            selected_trigger=None,
        )
        result = service.process_state(healthy_state, force=True)
        assert result is None

    def test_process_state_force_bypasses_interval(
        self, service: VoiceCoachService, healthy_state: GameState, mock_evaluate_usecase
    ):
        """Test process_state with force=True bypasses interval check."""
        service.last_response_ms = 0
        with patch("application.services.voice_coach.get_timestamp_ms", return_value=100):
            mock_evaluate_usecase.execute.return_value = MagicMock(
                has_firing_triggers=False,
                selected_trigger=None,
            )
            service.process_state(healthy_state, force=True)
            # Should have called execute even with short interval
            mock_evaluate_usecase.execute.assert_called_once()

    def test_mark_playback_complete_no_current(self, service: VoiceCoachService):
        """Test mark_playback_complete with no current response."""
        result = service.mark_playback_complete()
        assert result is None
        assert service.state == CoachState.IDLE

    def test_mark_playback_complete_with_queue(self, service: VoiceCoachService):
        """Test mark_playback_complete plays next from queue."""
        from application.dto.response_dto import AudioResponseDTO, QueuedResponseDTO

        # Set up current response
        current = AudioResponseDTO(text="current", priority=0)
        service.current_response = current
        service.state = CoachState.SPEAKING

        # Set up queue
        next_response = AudioResponseDTO(text="next", priority=0)
        service.response_queue = [
            QueuedResponseDTO(response=next_response, queue_position=0),
        ]

        with patch("application.services.voice_coach.get_timestamp_ms", return_value=1000):
            result = service.mark_playback_complete()

        assert result == next_response
        assert service.current_response == next_response
        assert service.state == CoachState.SPEAKING

    def test_queue_response(self, service: VoiceCoachService):
        """Test _queue_response adds to queue."""
        from application.dto.response_dto import AudioResponseDTO

        response = AudioResponseDTO(text="test", priority=0)
        result = service._queue_response(response)
        assert result is True
        assert len(service.response_queue) == 1

    def test_queue_response_full_queue(self, service: VoiceCoachService):
        """Test _queue_response returns False when queue full."""
        from application.dto.response_dto import AudioResponseDTO, QueuedResponseDTO

        # Fill queue
        service.response_queue = [
            QueuedResponseDTO(response=AudioResponseDTO(text=f"test{i}", priority=0), queue_position=i)
            for i in range(service.max_queue_size)
        ]

        # Try to add same priority
        response = AudioResponseDTO(text="new", priority=0)
        result = service._queue_response(response)
        assert result is False

    def test_interrupt_current(self, service: VoiceCoachService):
        """Test _interrupt_current clears current response."""
        from application.dto.response_dto import AudioResponseDTO

        response = AudioResponseDTO(text="test", priority=0)
        service.current_response = response
        service.state = CoachState.SPEAKING

        service._interrupt_current()

        assert service.current_response is None
        assert service.state == CoachState.IDLE
        assert response.interrupted is True


class TestVoiceCoachServiceQueuePriority:
    """Tests for VoiceCoachService queue priority handling."""

    @pytest.fixture
    def mock_evaluate_usecase(self):
        mock = MagicMock()
        mock.execute.return_value = MagicMock(has_firing_triggers=False, selected_trigger=None)
        return mock

    @pytest.fixture
    def mock_generate_usecase(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_evaluate_usecase, mock_generate_usecase) -> VoiceCoachService:
        return VoiceCoachService(
            evaluate_usecase=mock_evaluate_usecase,
            generate_usecase=mock_generate_usecase,
        )

    def test_queue_higher_priority_replaces_lower(self, service: VoiceCoachService):
        """Test higher priority response replaces lower in full queue."""
        from application.dto.response_dto import AudioResponseDTO, QueuedResponseDTO

        # Fill queue with low priority items
        service.response_queue = [
            QueuedResponseDTO(
                response=AudioResponseDTO(text=f"low{i}", priority=5),
                queue_position=i,
            )
            for i in range(service.max_queue_size)
        ]

        # Add high priority response
        high_priority = AudioResponseDTO(text="urgent", priority=0)
        result = service._queue_response(high_priority)

        assert result is True
        # Check that we still have max_queue_size items
        assert len(service.response_queue) == service.max_queue_size
