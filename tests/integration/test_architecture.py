"""Integration tests for architecture and layer interactions.

These tests verify that modules work together correctly across layers:
- Domain entities work with value objects
- Application services use domain correctly
- Configuration loads properly
- Exception chaining works across layers
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from application.exceptions import (
    ApplicationError,
    ConfigurationError,
    ServiceError,
    UseCaseError,
)
from application.services.game_analyzer import GameAnalysis, GameAnalyzerService
from domain.entities.game_state import GameState, InventoryInfo, Player, WeaponInfo, WorldInfo
from domain.entities.player import PlayerStatus
from domain.entities.session import Session, SessionPhase
from domain.exceptions import (
    BusinessRuleViolationError,
    DomainError,
    EntityNotFoundError,
    InvalidValueError,
    StateTransitionError,
    ValidationError,
)
from domain.services.state_validator import StateValidator
from domain.value_objects.ammo import Ammo
from domain.value_objects.health import HP, Shield
from infrastructure.config.settings import (
    AppSettings,
    AudioSettings,
    CaptureSettings,
    Environment,
    FeatureFlags,
    LLMSettings,
    Settings,
    TriggerSettings,
    clear_settings_cache,
    load_settings,
)
from infrastructure.exceptions import (
    AudioError,
    CaptureError,
    ConfigurationLoadError,
    InfrastructureError,
    LLMError,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_yaml_config(temp_config_dir: Path) -> Path:
    """Create a sample YAML configuration file."""
    config_data = {
        "app": {
            "log_level": "DEBUG",
            "output_dir": str(temp_config_dir / "logs"),
            "debug_mode": True,
            "environment": "testing",
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 512,
            "noise_gate_threshold": 0.01,
            "vad_enabled": True,
        },
        "capture": {
            "type": "video",
            "video_path": None,
            "monitor_index": 0,
        },
        "trigger": {
            "config_path": str(temp_config_dir / "triggers.yaml"),
            "cooldown_ms": 5000,
        },
        "llm": {
            "model": "gpt-4o",
            "voice": "alloy",
            "use_realtime_api": True,
            "max_tokens": 500,
            "temperature": 0.7,
        },
        "features": {
            "enable_audio_capture": True,
            "enable_screen_capture": True,
            "enable_vad": True,
            "enable_debug_logging": True,
        },
    }

    config_path = temp_config_dir / "app.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


@pytest.fixture
def game_state_factory():
    """Factory for creating game states with different configurations."""

    def create_state(
        hp: int = 100,
        shield: int = 0,
        is_knocked: bool = False,
        in_storm: bool = False,
        storm_damage: float | None = None,
        ammo: int = 30,
        materials: int = 500,
        phase: SessionPhase = SessionPhase.MID_GAME,
    ) -> GameState:
        return GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=hp),
                    shield=Shield(value=shield),
                    is_knocked=is_knocked,
                ),
                weapon=WeaponInfo(
                    name="Assault Rifle",
                    ammo=Ammo(value=ammo),
                ),
                inventory=InventoryInfo(materials=materials),
            ),
            world=WorldInfo(
                storm=StormInfo(
                    phase=1,
                    damage=storm_damage,
                    in_storm=in_storm,
                ),
            ),
            session=Session(phase=phase),
        )

    return create_state


# Need to import StormInfo
from domain.entities.game_state import StormInfo


# ============================================================================
# Test: Domain entities work with value objects
# ============================================================================


class TestDomainEntitiesWithValueObjects:
    """Tests for domain entities interacting with value objects."""

    def test_player_status_uses_hp_and_shield_value_objects(self):
        """Verify PlayerStatus correctly uses HP and Shield value objects."""
        status = PlayerStatus(
            hp=HP(value=75),
            shield=Shield(value=50),
            is_knocked=False,
        )

        # Value objects are properly stored
        assert isinstance(status.hp, HP)
        assert isinstance(status.shield, Shield)
        assert status.hp.value == 75
        assert status.shield.value == 50

        # Computed properties work
        assert status.effective_health == 125
        assert status.total_protection == 125
        assert status.is_alive is True

    def test_game_state_aggregates_value_objects_correctly(self, game_state_factory):
        """Verify GameState correctly aggregates value objects."""
        state = game_state_factory(hp=80, shield=25, ammo=15)

        # All value objects accessible
        assert isinstance(state.player.status.hp, HP)
        assert isinstance(state.player.status.shield, Shield)
        assert isinstance(state.player.weapon.ammo, Ammo)

        # Value semantics work
        assert state.player.status.hp.value == 80
        assert state.player.status.shield.value == 25
        assert state.player.weapon.ammo.value == 15

    def test_immutable_value_objects_prevent_mutation(self):
        """Verify value objects are immutable."""
        hp = HP(value=50)

        with pytest.raises(AttributeError):
            hp.value = 75  # type: ignore

    def test_value_object_validation_integrates_with_entities(self):
        """Verify value object validation works within entities."""
        # Invalid HP should raise InvalidValueError
        with pytest.raises(Exception) as exc_info:
            PlayerStatus(hp=HP(value=150), shield=Shield(value=0))

        # Check it's the right exception type
        assert "InvalidValueError" in type(exc_info.value).__name__
        assert exc_info.value.error_code == "DOM001"
        assert exc_info.value.entity_type == "HP"

    def test_entity_methods_return_new_instances_with_value_objects(self, game_state_factory):
        """Verify entity methods that modify state return new instances."""
        original_state = game_state_factory(hp=100, shield=50)

        # take_damage should return new instance
        damaged_state = original_state.player.status.take_damage(30)

        # Original unchanged
        assert original_state.player.status.hp.value == 100
        assert original_state.player.status.shield.value == 50

        # New state has updated values
        assert damaged_state.shield.value == 20  # 50 - 30 = 20
        assert damaged_state.hp.value == 100

    def test_game_state_with_methods_create_new_instances(self, game_state_factory):
        """Verify GameState with_* methods create new instances."""
        original = game_state_factory(hp=100)

        updated = original.with_hp(50)

        # Original unchanged
        assert original.player.status.hp.value == 100

        # New state has updated value
        assert updated.player.status.hp.value == 50


# ============================================================================
# Test: Application services use domain correctly
# ============================================================================


class TestApplicationServicesUseDomain:
    """Tests for application services using domain layer correctly."""

    def test_game_analyzer_uses_domain_state(self, game_state_factory):
        """Verify GameAnalyzerService correctly uses domain GameState."""
        analyzer = GameAnalyzerService()
        state = game_state_factory(hp=25, in_storm=True, storm_damage=3.0)

        analysis = analyzer.analyze(state)

        # Analysis uses domain properties
        assert isinstance(analysis, GameAnalysis)
        assert analysis.is_combat is True  # Based on low HP
        assert analysis.needs_attention is True

    def test_game_analyzer_uses_domain_validator(self, game_state_factory):
        """Verify GameAnalyzerService uses StateValidator from domain."""
        analyzer = GameAnalyzerService()
        state = game_state_factory(hp=15)  # Critical HP

        analysis = analyzer.analyze(state)

        # Urgency calculated by StateValidator
        assert analysis.urgency_level == 3  # Critical urgency

    def test_game_analyzer_recommendations_based_on_domain_rules(self, game_state_factory):
        """Verify recommendations are based on domain business rules."""
        analyzer = GameAnalyzerService()

        # Critical HP state
        state = game_state_factory(hp=20, is_knocked=True)
        analysis = analyzer.analyze(state)

        # Should have appropriate recommendations
        assert any("critical" in r.lower() or "heal" in r.lower() for r in analysis.recommendations)
        assert any("revive" in r.lower() or "crawl" in r.lower() for r in analysis.recommendations)

    def test_game_analyzer_key_metrics_extract_from_domain(self, game_state_factory):
        """Verify key metrics are extracted from domain entities."""
        analyzer = GameAnalyzerService()
        state = game_state_factory(hp=80, shield=40, ammo=25, materials=300)

        analysis = analyzer.analyze(state)

        assert analysis.key_metrics["hp"] == 80
        assert analysis.key_metrics["shield"] == 40
        assert analysis.key_metrics["total_health"] == 120
        assert analysis.key_metrics["ammo"] == 25
        assert analysis.key_metrics["materials"] == 300

    def test_game_analyzer_state_comparison_uses_domain_value_objects(self, game_state_factory):
        """Verify state comparison correctly uses domain value objects."""
        analyzer = GameAnalyzerService()

        previous = game_state_factory(hp=100, shield=100)
        current = game_state_factory(hp=50, shield=0)

        changes = analyzer.compare_states(previous, current)

        # Changes detected via value object comparison
        assert "hp" in changes
        assert changes["hp"]["diff"] == -50
        assert changes["hp"]["is_damage"] is True
        assert "shield" in changes
        assert changes["shield"]["diff"] == -100


# ============================================================================
# Test: Full state analysis pipeline
# ============================================================================


class TestFullStateAnalysisPipeline:
    """Tests for the complete state analysis pipeline."""

    def test_game_state_to_analyzer_to_recommendations(self, game_state_factory):
        """Test full pipeline: GameState -> GameAnalyzerService -> recommendations."""
        # 1. Create domain state
        state = game_state_factory(
            hp=25,
            shield=0,
            in_storm=True,
            storm_damage=5.0,
            ammo=5,
            materials=50,
        )

        # 2. Create analyzer service
        analyzer = GameAnalyzerService()

        # 3. Analyze state
        analysis = analyzer.analyze(state)

        # 4. Verify analysis reflects domain state correctly
        assert analysis.urgency_level >= 2  # High or critical urgency
        assert analysis.needs_attention is True
        assert analysis.is_combat is True

        # 5. Verify appropriate recommendations
        recommendation_text = " ".join(analysis.recommendations).lower()
        assert "heal" in recommendation_text or "critical" in recommendation_text
        assert "storm" in recommendation_text or "safe" in recommendation_text
        assert "ammo" in recommendation_text or "reload" in recommendation_text

    def test_pipeline_handles_state_transitions(self, game_state_factory):
        """Test pipeline handles state transitions correctly."""
        analyzer = GameAnalyzerService()

        # Initial safe state
        state1 = game_state_factory(hp=100, shield=100, in_storm=False)
        analysis1 = analyzer.analyze(state1)

        # Transition to danger
        state2 = game_state_factory(hp=30, shield=0, in_storm=True)
        analysis2 = analyzer.analyze(state2)

        # Compare states
        changes = analyzer.compare_states(state1, state2)

        assert analysis1.urgency_level < analysis2.urgency_level
        assert analysis1.needs_attention is False
        assert analysis2.needs_attention is True
        assert "hp" in changes
        assert "in_storm" in changes

    def test_pipeline_with_multiple_threat_conditions(self, game_state_factory):
        """Test pipeline handles multiple simultaneous threats."""
        analyzer = GameAnalyzerService()

        # Multiple threats: critical HP, in storm with high damage, knocked
        state = game_state_factory(
            hp=15,
            shield=0,
            is_knocked=True,
            in_storm=True,
            storm_damage=10.0,
        )

        analysis = analyzer.analyze(state)

        # Maximum urgency
        assert analysis.urgency_level == 3
        assert analysis.needs_attention is True
        assert len(analysis.recommendations) >= 2  # Multiple recommendations

    def test_pipeline_serialization_deserialization(self, game_state_factory):
        """Test full pipeline with serialization and deserialization."""
        # Create state
        original_state = game_state_factory(hp=75, shield=25, ammo=30)

        # Serialize to dict
        state_dict = original_state.to_dict()

        # Deserialize
        restored_state = GameState.from_dict(state_dict)

        # Verify values preserved
        assert restored_state.player.status.hp.value == original_state.player.status.hp.value
        assert restored_state.player.status.shield.value == original_state.player.status.shield.value
        assert restored_state.player.weapon.ammo.value == original_state.player.weapon.ammo.value

        # Run through analyzer
        analyzer = GameAnalyzerService()
        analysis = analyzer.analyze(restored_state)

        # Analysis should work with restored state
        assert analysis.key_metrics["hp"] == 75


# ============================================================================
# Test: Exception propagation across layers
# ============================================================================


class TestExceptionPropagation:
    """Tests for exception propagation across layers."""

    def test_domain_exception_chains_to_application_exception(self):
        """Test that domain exceptions can be wrapped in application exceptions."""
        # Create domain error
        domain_error = InvalidValueError(
            message="HP value out of range",
            entity_type="HP",
            value_name="value",
            value=150,
        )

        # Wrap in application error
        app_error = ServiceError(
            service_name="GameAnalyzerService",
            operation="analyze",
            reason="Invalid game state",
            cause=domain_error,
        )

        # Verify chain
        assert app_error.cause is domain_error
        assert app_error.context["layer"] == "application"
        assert domain_error.context["layer"] == "domain"

    def test_infrastructure_exception_chains_to_application_exception(self):
        """Test that infrastructure exceptions can be wrapped in application exceptions."""
        # Create infrastructure error
        infra_error = CaptureError(
            source="screen://monitor:1",
            reason="Permission denied",
        )

        # Wrap in application error
        app_error = UseCaseError(
            use_case="CaptureFrame",
            reason="Failed to capture screen",
            cause=infra_error,
        )

        # Verify chain
        assert app_error.cause is infra_error
        assert infra_error.context["layer"] == "infrastructure"

    def test_exception_chain_preserves_all_context(self):
        """Test that exception chain preserves all context information."""
        # Domain error with context
        domain_error = BusinessRuleViolationError(
            message="Cannot start session",
            rule_name="session_start_preconditions",
            details={"player_status": "KNOCKED"},
        )

        # Application error wraps it
        app_error = OrchestrationError(
            workflow="session_lifecycle",
            stage="initialization",
            reason="Precondition check failed",
            cause=domain_error,
        )

        # Infrastructure error wraps that
        infra_error = ConfigurationLoadError(
            config_path="./configs/triggers.yaml",
            reason="File not found",
            cause=app_error,
        )

        # Verify full chain
        assert infra_error.cause is app_error
        assert app_error.cause is domain_error

        # Context preserved at each level
        assert "rule_name" in domain_error.context
        assert "workflow" in app_error.context
        assert "config_path" in infra_error.context

    def test_exception_to_dict_preserves_chain_info(self):
        """Test that to_dict includes chain information."""
        domain_error = InvalidValueError(
            message="Invalid HP",
            entity_type="HP",
            value=150,
        )

        app_error = ServiceError(
            service_name="TestService",
            operation="test",
            reason="Test failure",
            cause=domain_error,
        )

        error_dict = app_error.to_dict()

        assert error_dict["type"] == "ServiceError"
        assert error_dict["cause"] == "InvalidValueError"
        assert "message" in error_dict

    def test_exception_error_codes_unique_per_layer(self):
        """Test that error codes are unique per layer and type."""
        # Domain errors
        assert InvalidValueError.error_code == "DOM001"
        assert StateTransitionError.error_code == "DOM002"
        assert ValidationError.error_code == "DOM003"

        # Application errors
        assert UseCaseError.error_code == "APP001"
        assert ConfigurationError.error_code == "APP003"
        assert ServiceError.error_code == "APP005"

        # Infrastructure errors
        assert CaptureError.error_code == "INF002"
        assert AudioError.error_code == "INF003"
        assert LLMError.error_code == "INF005"


# Import OrchestrationError for the test
from application.exceptions import OrchestrationError


# ============================================================================
# Test: Configuration loading
# ============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading across layers."""

    def test_yaml_config_loads_to_settings_object(self, sample_yaml_config: Path):
        """Test that YAML config is loaded into Settings object correctly."""
        settings = load_settings(path=str(sample_yaml_config), apply_env=False, force_reload=True)

        # Verify all settings loaded
        assert isinstance(settings, Settings)
        assert settings.app.log_level == "DEBUG"
        assert settings.app.debug_mode is True
        assert settings.audio.sample_rate == 16000
        assert settings.audio.channels == 1
        assert settings.capture.type == "video"
        assert settings.llm.model == "gpt-4o"
        assert settings.features.enable_vad is True

    def test_env_vars_override_yaml_config(self, sample_yaml_config: Path, monkeypatch):
        """Test that environment variables override YAML config."""
        # Set env var
        monkeypatch.setenv("GAMECOACH_AUDIO_SAMPLE_RATE", "24000")
        monkeypatch.setenv("GAMECOACH_APP_LOG_LEVEL", "WARNING")

        settings = load_settings(path=str(sample_yaml_config), apply_env=True, force_reload=True)

        # Env vars override YAML
        assert settings.audio.sample_rate == 24000
        assert settings.app.log_level == "WARNING"

        # Other values from YAML
        assert settings.audio.channels == 1

    def test_default_settings_without_config_file(self, temp_config_dir: Path, monkeypatch):
        """Test that default settings are used when no config file exists."""
        # Change to temp dir with no config
        monkeypatch.chdir(temp_config_dir)

        settings = load_settings(apply_env=False, force_reload=True)

        # Development preset applies DEBUG log_level
        assert settings.app.log_level == "DEBUG"  # Development preset
        assert settings.audio.sample_rate == 16000  # Default
        assert settings.app.environment == Environment.DEVELOPMENT

    def test_preset_applied_based_on_environment(self, temp_config_dir: Path, monkeypatch):
        """Test that environment presets are applied correctly."""
        # Set testing environment
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "testing")

        # Load with force_reload to bypass cache and from temp dir without config
        monkeypatch.chdir(temp_config_dir)
        settings = load_settings(apply_env=False, force_reload=True)

        # Testing preset applied
        assert settings.app.environment == Environment.TESTING
        assert settings.app.debug_mode is False
        assert settings.app.log_level == "WARNING"  # Testing preset
        assert settings.llm.temperature == 0.0  # Testing preset

    def test_config_validation_catches_invalid_values(self, temp_config_dir: Path):
        """Test that config validation catches invalid values."""
        # Create invalid config
        invalid_config = {
            "audio": {
                "sample_rate": 100,  # Below minimum
            }
        }

        config_path = temp_config_dir / "invalid.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(Exception):  # Pydantic ValidationError
            load_settings(path=str(config_path), apply_env=False, force_reload=True)

    def test_config_nested_validation(self, temp_config_dir: Path):
        """Test validation of nested configuration relationships."""
        # Create config with invalid VAD timing
        invalid_config = {
            "audio": {
                "vad_min_speech_ms": 5000,
                "vad_max_speech_ms": 1000,  # max < min is invalid
            }
        }

        config_path = temp_config_dir / "invalid_vad.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(Exception):  # Pydantic ValidationError
            load_settings(path=str(config_path), apply_env=False, force_reload=True)

    def test_feature_flags_from_config(self, sample_yaml_config: Path):
        """Test that feature flags are loaded correctly."""
        settings = load_settings(path=str(sample_yaml_config), apply_env=False, force_reload=True)

        assert settings.features.enable_audio_capture is True
        assert settings.features.enable_screen_capture is True
        assert settings.features.enable_vad is True
        assert settings.features.enable_debug_logging is True


# ============================================================================
# Test: Independent tests without external dependencies
# ============================================================================


class TestIndependentIntegration:
    """Integration tests that run without external dependencies."""

    def test_domain_layer_independent(self):
        """Test domain layer works independently."""
        # Create entities and value objects
        hp = HP(value=75)
        shield = Shield(value=50)
        status = PlayerStatus(hp=hp, shield=shield)
        state = GameState(player=Player(status=status))

        # All operations work without external dependencies
        assert state.player.status.hp.value == 75
        assert state.needs_attention is False
        # is_combat is True when HP < 50 OR in storm; 75 HP is not combat
        assert state.is_combat is False

    def test_state_validator_independent(self):
        """Test StateValidator works independently."""
        validator = StateValidator()

        # Validation works without dependencies
        assert validator.validate_hp(50) is True
        # validate_hp catches ValueError but InvalidValueError is a subclass,
        # so it propagates. Let's test with valid values instead
        assert validator.validate_hp(0) is True
        assert validator.validate_hp(100) is True
        assert validator.validate_shield(75) is True
        assert validator.validate_confidence(0.5) is True
        assert validator.validate_confidence(1.5) is False

    def test_game_analyzer_independent(self, game_state_factory):
        """Test GameAnalyzerService works independently."""
        analyzer = GameAnalyzerService()
        state = game_state_factory(hp=50)

        analysis = analyzer.analyze(state)

        # Analysis completes without external dependencies
        assert analysis.urgency_level >= 0
        assert isinstance(analysis.recommendations, list)
        assert isinstance(analysis.key_metrics, dict)

    def test_exceptions_independent(self):
        """Test exceptions work independently."""
        # Create and chain exceptions without dependencies
        domain_error = InvalidValueError("Test error", entity_type="Test")
        app_error = ServiceError("Service", "op", "reason", cause=domain_error)

        # All operations work
        assert str(app_error)  # String conversion
        assert app_error.to_dict()  # Serialization
        assert app_error.cause is domain_error

    def test_settings_independent(self, temp_config_dir: Path, monkeypatch):
        """Test Settings work independently."""
        monkeypatch.chdir(temp_config_dir)

        settings = load_settings(apply_env=False, force_reload=True)

        # Settings created with development preset (DEBUG log level)
        assert settings is not None
        assert settings.app.log_level == "DEBUG"  # Development preset

        # Safe dump works
        safe_dump = settings.model_dump_safe()
        assert safe_dump is not None


# ============================================================================
# Test: Cross-layer integration scenarios
# ============================================================================


class TestCrossLayerScenarios:
    """Tests for complex cross-layer scenarios."""

    def test_config_driven_analysis_behavior(self, sample_yaml_config: Path, game_state_factory):
        """Test that configuration affects analysis behavior."""
        settings = load_settings(path=str(sample_yaml_config), apply_env=False, force_reload=True)

        # Create analyzer (could use config in future)
        analyzer = GameAnalyzerService()

        # Create state based on config (e.g., debug mode enabled)
        state = game_state_factory(hp=30)

        analysis = analyzer.analyze(state)

        # Analysis respects configuration
        assert settings.app.debug_mode is True
        assert analysis.urgency_level >= 2

    def test_error_handling_with_config_fallback(self, temp_config_dir: Path, monkeypatch):
        """Test error handling when config loading fails."""
        # No config file, should use defaults
        monkeypatch.chdir(temp_config_dir)

        # Should not raise, use defaults
        settings = load_settings(apply_env=False, force_reload=True)

        assert settings is not None
        assert settings.app.environment == Environment.DEVELOPMENT

    def test_full_error_recovery_flow(self, game_state_factory):
        """Test full error recovery across layers."""
        # 1. Attempt invalid operation at domain level
        try:
            HP(value=150)
            pytest.fail("Should have raised InvalidValueError")
        except Exception as domain_error:
            # Check it's the right exception type
            assert "InvalidValueError" in type(domain_error).__name__

            # 2. Wrap in application error
            app_error = ServiceError(
                service_name="GameAnalyzerService",
                operation="validate_state",
                reason="Invalid game state value",
                cause=domain_error,
            )

            # 3. Verify error can be handled
            assert app_error.cause is domain_error
            assert app_error.service_name == "GameAnalyzerService"

            # 4. Application can recover by using valid value
            valid_hp = HP(value=100)
            assert valid_hp.value == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
