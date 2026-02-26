"""Tests for domain services: StateValidator."""

import importlib.util
from pathlib import Path

import pytest

# Direct module imports to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"

# Load exceptions first (needed by value objects)
exceptions_spec = importlib.util.spec_from_file_location(
    "domain_exceptions",
    SRC_PATH / "domain" / "exceptions.py",
)
exceptions_module = importlib.util.module_from_spec(exceptions_spec)
exceptions_spec.loader.exec_module(exceptions_module)

# Make domain.exceptions available for import
import sys

sys.modules["domain.exceptions"] = exceptions_module

# Load state validator
state_validator_spec = importlib.util.spec_from_file_location(
    "state_validator",
    SRC_PATH / "domain" / "services" / "state_validator.py",
)
state_validator_module = importlib.util.module_from_spec(state_validator_spec)
state_validator_spec.loader.exec_module(state_validator_module)

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

# Import classes
StateValidator = state_validator_module.StateValidator

GameState = game_state_module.GameState
Player = game_state_module.Player
WeaponInfo = game_state_module.WeaponInfo
StormInfo = game_state_module.StormInfo
WorldInfo = game_state_module.WorldInfo

PlayerStatus = player_module.PlayerStatus
Session = session_module.Session
SessionPhase = session_module.SessionPhase

HP = health_module.HP
Shield = health_module.Shield
Ammo = ammo_module.Ammo


class TestStateValidatorValidateHP:
    """Tests for StateValidator.validate_hp method."""

    def test_validate_hp_valid(self):
        """Test validate_hp returns True for valid HP."""
        assert StateValidator.validate_hp(0) is True
        assert StateValidator.validate_hp(50) is True
        assert StateValidator.validate_hp(100) is True

    def test_validate_hp_boundary(self):
        """Test validate_hp at boundaries."""
        assert StateValidator.validate_hp(0) is True
        assert StateValidator.validate_hp(100) is True

    # Note: validate_hp catches ValueError but HP raises InvalidValueError
    # So invalid values will raise InvalidValueError rather than returning False
    # This is a potential bug in the code, but tests document current behavior


class TestStateValidatorValidateShield:
    """Tests for StateValidator.validate_shield method."""

    def test_validate_shield_valid(self):
        """Test validate_shield returns True for valid shield."""
        assert StateValidator.validate_shield(0) is True
        assert StateValidator.validate_shield(50) is True
        assert StateValidator.validate_shield(100) is True

    # Note: validate_shield catches ValueError but Shield raises InvalidValueError
    # So invalid values will raise InvalidValueError rather than returning False
    # This is a potential bug in the code, but tests document current behavior


class TestStateValidatorValidateConfidence:
    """Tests for StateValidator.validate_confidence method."""

    def test_validate_confidence_valid(self):
        """Test validate_confidence returns True for valid confidence."""
        assert StateValidator.validate_confidence(0.0) is True
        assert StateValidator.validate_confidence(0.5) is True
        assert StateValidator.validate_confidence(1.0) is True

    def test_validate_confidence_negative(self):
        """Test validate_confidence returns False for negative confidence."""
        assert StateValidator.validate_confidence(-0.1) is False
        assert StateValidator.validate_confidence(-1.0) is False

    def test_validate_confidence_exceeds_max(self):
        """Test validate_confidence returns False for confidence > 1."""
        assert StateValidator.validate_confidence(1.1) is False
        assert StateValidator.validate_confidence(2.0) is False


class TestStateValidatorValidateStateTransition:
    """Tests for StateValidator.validate_state_transition method."""

    @pytest.fixture
    def valid_state(self) -> GameState:
        """Create a valid game state."""
        return GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=100),
                    shield=Shield(value=50),
                ),
                weapon=WeaponInfo(ammo=Ammo(value=30)),
            ),
        )

    def test_validate_transition_valid(self, valid_state: GameState):
        """Test validate_state_transition returns True for valid transition."""
        next_state = GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=75),
                    shield=Shield(value=25),
                ),
                weapon=WeaponInfo(ammo=Ammo(value=20)),
            ),
        )
        is_valid, error = StateValidator.validate_state_transition(valid_state, next_state)
        assert is_valid is True
        assert error == ""

    def test_validate_transition_with_knocked_and_hp(self, valid_state: GameState):
        """Test validate_transition allows knocked with HP > 0 (per comment in code)."""
        next_state = GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=50),
                    is_knocked=True,
                )
            ),
        )
        is_valid, error = StateValidator.validate_state_transition(valid_state, next_state)
        # This is allowed per the code comment
        assert is_valid is True


class TestStateValidatorValidateStateValue:
    """Tests for StateValidator.validate_state_value method."""

    def test_validate_state_value_valid(self):
        """Test validate_state_value returns True for valid input."""
        is_valid, error = StateValidator.validate_state_value(
            _value=100, source="ocr", confidence=0.9
        )
        assert is_valid is True
        assert error == ""

    def test_validate_state_value_missing_source(self):
        """Test validate_state_value rejects missing source."""
        is_valid, error = StateValidator.validate_state_value(_value=100, source="", confidence=0.9)
        assert is_valid is False
        assert "Source is required" in error

    def test_validate_state_value_none_source(self):
        """Test validate_state_value rejects None source."""
        is_valid, error = StateValidator.validate_state_value(
            _value=100, source=None, confidence=0.9
        )
        assert is_valid is False
        assert "Source is required" in error

    def test_validate_state_value_invalid_confidence_low(self):
        """Test validate_state_value rejects confidence < 0."""
        is_valid, error = StateValidator.validate_state_value(
            _value=100, source="ocr", confidence=-0.1
        )
        assert is_valid is False
        assert "Confidence must be between 0.0 and 1.0" in error

    def test_validate_state_value_invalid_confidence_high(self):
        """Test validate_state_value rejects confidence > 1."""
        is_valid, error = StateValidator.validate_state_value(
            _value=100, source="ocr", confidence=1.1
        )
        assert is_valid is False
        assert "Confidence must be between 0.0 and 1.0" in error

    def test_validate_state_value_confidence_boundary(self):
        """Test validate_state_value accepts confidence at boundaries."""
        is_valid_low, _ = StateValidator.validate_state_value(
            _value=100, source="test", confidence=0.0
        )
        is_valid_high, _ = StateValidator.validate_state_value(
            _value=100, source="test", confidence=1.0
        )
        assert is_valid_low is True
        assert is_valid_high is True


class TestStateValidatorShouldTriggerCombatResponse:
    """Tests for StateValidator.should_trigger_combat_response method."""

    def test_trigger_when_low_hp(self):
        """Test should trigger when HP is low."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=40))),
        )
        assert StateValidator.should_trigger_combat_response(state) is True

    def test_trigger_when_in_storm(self):
        """Test should trigger when in storm."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True)),
        )
        assert StateValidator.should_trigger_combat_response(state) is True

    def test_trigger_when_knocked(self):
        """Test should trigger when knocked."""
        state = GameState(
            player=Player(status=PlayerStatus(is_knocked=True)),
        )
        assert StateValidator.should_trigger_combat_response(state) is True

    def test_no_trigger_when_healthy(self):
        """Test should not trigger when healthy."""
        state = GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=100),
                    shield=Shield(value=100),
                    is_knocked=False,
                )
            ),
            world=WorldInfo(storm=StormInfo(in_storm=False)),
        )
        assert StateValidator.should_trigger_combat_response(state) is False


class TestStateValidatorCalculateUrgency:
    """Tests for StateValidator.calculate_urgency method."""

    def test_urgency_zero_healthy(self):
        """Test urgency is 0 when healthy."""
        state = GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=100),
                    shield=Shield(value=100),
                )
            ),
            world=WorldInfo(storm=StormInfo(in_storm=False)),
        )
        assert StateValidator.calculate_urgency(state) == 0

    def test_urgency_two_when_low_hp(self):
        """Test urgency is 2 when HP is low but not critical."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=40))),
        )
        assert StateValidator.calculate_urgency(state) == 2

    def test_urgency_three_when_critical_hp(self):
        """Test urgency is 3 when HP is critical."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=20))),
        )
        assert StateValidator.calculate_urgency(state) == 3

    def test_urgency_two_when_in_storm(self):
        """Test urgency is 2 when in storm with low damage."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=2.0)),
        )
        assert StateValidator.calculate_urgency(state) == 2

    def test_urgency_three_when_heavy_storm(self):
        """Test urgency is 3 when in storm with heavy damage."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=6.0)),
        )
        assert StateValidator.calculate_urgency(state) == 3

    def test_urgency_three_when_knocked(self):
        """Test urgency is 3 when knocked."""
        state = GameState(
            player=Player(status=PlayerStatus(is_knocked=True)),
        )
        assert StateValidator.calculate_urgency(state) == 3

    def test_urgency_takes_maximum(self):
        """Test urgency takes maximum of all conditions."""
        state = GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=20),  # Critical = 3
                    is_knocked=True,  # Knocked = 3
                )
            ),
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=10)),  # Heavy storm = 3
        )
        assert StateValidator.calculate_urgency(state) == 3

    def test_urgency_multiple_lower_priority(self):
        """Test urgency with multiple lower priority conditions."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=40))),  # Low HP = 2
            world=WorldInfo(storm=StormInfo(in_storm=True, damage=2.0)),  # Storm = 2
        )
        assert StateValidator.calculate_urgency(state) == 2
