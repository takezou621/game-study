"""Tests for infrastructure configuration: settings loading."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestEnvironment:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test Environment enum values."""
        from src.infrastructure.config.settings import Environment
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.PRODUCTION.value == "production"


class TestFeatureFlags:
    """Tests for FeatureFlags class."""

    def test_init_defaults(self):
        """Test FeatureFlags initialization with defaults."""
        from src.infrastructure.config.settings import FeatureFlags
        flags = FeatureFlags()
        assert flags.enable_audio_capture is True
        assert flags.enable_screen_capture is True
        assert flags.enable_video_capture is True
        assert flags.enable_realtime_api is True
        assert flags.enable_vad is True
        assert flags.enable_noise_gate is True
        assert flags.enable_debug_logging is False
        assert flags.enable_performance_metrics is False

    def test_init_with_values(self):
        """Test FeatureFlags initialization with values."""
        from src.infrastructure.config.settings import FeatureFlags
        flags = FeatureFlags(
            enable_audio_capture=False,
            enable_debug_logging=True,
        )
        assert flags.enable_audio_capture is False
        assert flags.enable_debug_logging is True


class TestAppSettings:
    """Tests for AppSettings class."""

    def test_init_defaults(self):
        """Test AppSettings initialization with defaults."""
        from src.infrastructure.config.settings import AppSettings, Environment
        settings = AppSettings()
        assert settings.log_level == "INFO"
        assert settings.output_dir == "./logs"
        assert settings.debug_mode is False
        assert settings.environment == Environment.DEVELOPMENT

    def test_init_with_values(self):
        """Test AppSettings initialization with values."""
        from src.infrastructure.config.settings import AppSettings, Environment
        settings = AppSettings(
            log_level="DEBUG",
            output_dir="/var/log/app",
            debug_mode=True,
            environment=Environment.PRODUCTION,
        )
        assert settings.log_level == "DEBUG"
        assert settings.output_dir == "/var/log/app"
        assert settings.debug_mode is True
        assert settings.environment == Environment.PRODUCTION

    def test_model_dump(self):
        """Test model_dump method."""
        from src.infrastructure.config.settings import AppSettings
        settings = AppSettings(log_level="WARNING", debug_mode=True)
        result = settings.model_dump()
        assert result["log_level"] == "WARNING"
        assert result["debug_mode"] is True

    def test_expand_user_output_dir(self):
        """Test output_dir expands user home directory."""
        from src.infrastructure.config.settings import AppSettings
        settings = AppSettings(output_dir="~/logs")
        assert "~" not in settings.output_dir


class TestAudioSettings:
    """Tests for AudioSettings class."""

    def test_init_defaults(self):
        """Test AudioSettings initialization with defaults."""
        from src.infrastructure.config.settings import AudioSettings
        settings = AudioSettings()
        assert settings.sample_rate == 16000
        assert settings.channels == 1
        assert settings.chunk_size == 512
        assert settings.noise_gate_threshold == 0.01
        assert settings.vad_enabled is True
        assert settings.device_index is None

    def test_init_with_values(self):
        """Test AudioSettings initialization with values."""
        from src.infrastructure.config.settings import AudioSettings
        settings = AudioSettings(
            sample_rate=48000,
            channels=2,
            chunk_size=1024,
            device_index=1,
        )
        assert settings.sample_rate == 48000
        assert settings.channels == 2
        assert settings.chunk_size == 1024
        assert settings.device_index == 1

    def test_model_dump(self):
        """Test model_dump method."""
        from src.infrastructure.config.settings import AudioSettings
        settings = AudioSettings(sample_rate=44100)
        result = settings.model_dump()
        assert result["sample_rate"] == 44100
        assert "channels" in result
        assert "vad_enabled" in result

    def test_vad_timing_validation(self):
        """Test VAD timing validation."""
        from src.infrastructure.config.settings import AudioSettings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AudioSettings(
                vad_min_speech_ms=5000,
                vad_max_speech_ms=1000,  # Less than min
            )


class TestCaptureSettings:
    """Tests for CaptureSettings class."""

    def test_init_defaults(self):
        """Test CaptureSettings initialization with defaults."""
        from src.infrastructure.config.settings import CaptureSettings
        settings = CaptureSettings()
        assert settings.type == "video"
        assert settings.video_path is None
        assert settings.loop_video is False
        assert settings.monitor_index == 1
        assert settings.region is None

    def test_init_with_values(self):
        """Test CaptureSettings initialization with values."""
        from src.infrastructure.config.settings import CaptureSettings
        settings = CaptureSettings(
            type="screen",
            video_path="/path/to/video.mp4",
            loop_video=True,
            monitor_index=2,
            region=(0, 0, 1920, 1080),
        )
        assert settings.type == "screen"
        assert settings.video_path == "/path/to/video.mp4"
        assert settings.loop_video is True
        assert settings.monitor_index == 2
        assert settings.region == (0, 0, 1920, 1080)

    def test_model_dump(self):
        """Test model_dump method."""
        from src.infrastructure.config.settings import CaptureSettings
        settings = CaptureSettings(
            type="screen",
            monitor_index=2,
            region=(100, 100, 800, 600),
        )
        result = settings.model_dump()
        assert result["type"] == "screen"
        assert result["monitor_index"] == 2
        assert result["region"] == (100, 100, 800, 600)

    def test_region_from_list(self):
        """Test region conversion from list."""
        from src.infrastructure.config.settings import CaptureSettings
        settings = CaptureSettings(region=[0, 0, 1280, 720])
        assert settings.region == (0, 0, 1280, 720)

    def test_region_negative_raises(self):
        """Test region with negative values raises error."""
        from src.infrastructure.config.settings import CaptureSettings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CaptureSettings(region=(-1, 0, 100, 100))


class TestTriggerSettings:
    """Tests for TriggerSettings class."""

    def test_init_defaults(self):
        """Test TriggerSettings initialization with defaults."""
        from src.infrastructure.config.settings import TriggerSettings
        settings = TriggerSettings()
        assert settings.config_path == "./configs/triggers.yaml"
        assert settings.cooldown_ms == 5000
        assert settings.max_response_length_ms == 10000

    def test_init_with_values(self):
        """Test TriggerSettings initialization with values."""
        from src.infrastructure.config.settings import TriggerSettings
        settings = TriggerSettings(
            config_path="/custom/triggers.yaml",
            cooldown_ms=3000,
            max_response_length_ms=5000,
        )
        assert settings.config_path == "/custom/triggers.yaml"
        assert settings.cooldown_ms == 3000
        assert settings.max_response_length_ms == 5000

    def test_model_dump(self):
        """Test model_dump method."""
        from src.infrastructure.config.settings import TriggerSettings
        settings = TriggerSettings(cooldown_ms=2000)
        result = settings.model_dump()
        assert result["cooldown_ms"] == 2000


class TestLLMSettings:
    """Tests for LLMSettings class."""

    def test_init_defaults(self):
        """Test LLMSettings initialization with defaults."""
        from src.infrastructure.config.settings import LLMSettings
        settings = LLMSettings()
        assert settings.api_key is None
        assert settings.model == "gpt-4o"
        assert settings.voice == "alloy"
        assert settings.use_realtime_api is True
        assert settings.max_tokens == 500
        assert settings.temperature == 0.7

    def test_init_with_values(self):
        """Test LLMSettings initialization with values."""
        from src.infrastructure.config.settings import LLMSettings
        from pydantic import SecretStr
        settings = LLMSettings(
            api_key=SecretStr("test-key"),
            model="gpt-4-turbo",
            voice="nova",
            max_tokens=1000,
            temperature=0.5,
        )
        assert settings.get_api_key() == "test-key"
        assert settings.model == "gpt-4-turbo"
        assert settings.voice == "nova"
        assert settings.max_tokens == 1000
        assert settings.temperature == 0.5

    def test_model_dump_safe_excludes_api_key(self):
        """Test model_dump_safe masks api_key for security."""
        from src.infrastructure.config.settings import LLMSettings
        from pydantic import SecretStr
        settings = LLMSettings(api_key=SecretStr("secret-key"))
        result = settings.model_dump_safe()
        assert result["api_key"] == "***REDACTED***"

    def test_get_api_key_none(self):
        """Test get_api_key returns None when not set."""
        from src.infrastructure.config.settings import LLMSettings
        settings = LLMSettings()
        assert settings.get_api_key() is None


class TestSettings:
    """Tests for Settings aggregate class."""

    def test_init_defaults(self):
        """Test Settings initialization with defaults."""
        from src.infrastructure.config.settings import Settings
        from src.infrastructure.config.settings import AppSettings, AudioSettings, CaptureSettings, TriggerSettings, LLMSettings, FeatureFlags
        settings = Settings()
        assert isinstance(settings.app, AppSettings)
        assert isinstance(settings.audio, AudioSettings)
        assert isinstance(settings.capture, CaptureSettings)
        assert isinstance(settings.trigger, TriggerSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.features, FeatureFlags)

    def test_init_with_values(self):
        """Test Settings initialization with values."""
        from src.infrastructure.config.settings import Settings, AppSettings, AudioSettings
        settings = Settings(
            app=AppSettings(log_level="DEBUG"),
            audio=AudioSettings(sample_rate=48000),
        )
        assert settings.app.log_level == "DEBUG"
        assert settings.audio.sample_rate == 48000

    def test_model_dump(self):
        """Test model_dump method."""
        from src.infrastructure.config.settings import Settings, AppSettings, AudioSettings
        settings = Settings(
            app=AppSettings(log_level="DEBUG"),
            audio=AudioSettings(sample_rate=44100),
        )
        result = settings.model_dump()
        assert "app" in result
        assert "audio" in result
        assert "capture" in result
        assert "trigger" in result
        assert "llm" in result
        assert result["app"]["log_level"] == "DEBUG"
        assert result["audio"]["sample_rate"] == 44100

    def test_model_validate(self):
        """Test model_validate method."""
        from src.infrastructure.config.settings import Settings
        data = {
            "app": {"log_level": "WARNING"},
            "audio": {"sample_rate": 48000, "channels": 2},
            "capture": {"type": "screen"},
        }
        settings = Settings.model_validate(data)
        assert settings.app.log_level == "WARNING"
        assert settings.audio.sample_rate == 48000
        assert settings.audio.channels == 2
        assert settings.capture.type == "screen"


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings_from_path(self, tmp_path, monkeypatch):
        """Test load_settings from specific path."""
        from src.infrastructure.config.settings import load_settings, clear_settings_cache
        clear_settings_cache()
        # Set environment to testing to avoid preset overrides
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "testing")

        config_path = tmp_path / "custom_config.yaml"
        config_data = {
            "app": {"log_level": "DEBUG"},
            "audio": {"sample_rate": 48000},
        }
        config_path.write_text(yaml.dump(config_data))

        settings = load_settings(str(config_path), force_reload=True)
        assert settings.app.log_level == "DEBUG"
        assert settings.audio.sample_rate == 48000

    def test_load_settings_empty_file(self, tmp_path, monkeypatch):
        """Test load_settings handles empty file."""
        from src.infrastructure.config.settings import load_settings, clear_settings_cache
        clear_settings_cache()
        # Set environment to testing
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "testing")

        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        settings = load_settings(str(config_path), force_reload=True)
        # Should use defaults (testing preset has WARNING log level)
        assert settings.app.log_level == "WARNING"


class TestSaveSettings:
    """Tests for save_settings function."""

    def test_save_settings_creates_file(self, tmp_path, monkeypatch):
        """Test save_settings creates file."""
        from src.infrastructure.config.settings import Settings, save_settings, clear_settings_cache, Environment
        clear_settings_cache()
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "testing")

        settings = Settings()
        config_path = tmp_path / "output" / "config.yaml"
        save_settings(settings, str(config_path))

        assert config_path.exists()
        assert config_path.parent.is_dir()

    def test_save_settings_creates_parent_dirs(self, tmp_path):
        """Test save_settings creates parent directories."""
        from src.infrastructure.config.settings import Settings, save_settings
        settings = Settings()
        config_path = tmp_path / "nested" / "deep" / "config.yaml"
        save_settings(settings, str(config_path))

        assert config_path.exists()
        assert config_path.parent.is_dir()

    def test_save_settings_excludes_secrets_by_default(self, tmp_path):
        """Test save_settings does not save secrets by default."""
        from src.infrastructure.config.settings import Settings, LLMSettings, save_settings
        from pydantic import SecretStr
        settings = Settings(
            llm=LLMSettings(api_key=SecretStr("secret-key"), model="gpt-4"),
        )

        config_path = tmp_path / "secure.yaml"
        save_settings(settings, str(config_path), include_secrets=False)

        with open(config_path) as f:
            content = f.read()

        assert "secret-key" not in content


class TestSettingsHelperFunctions:
    """Tests for settings helper functions."""

    def test_clear_settings_cache(self):
        """Test clear_settings_cache function."""
        from src.infrastructure.config.settings import clear_settings_cache, _settings_cache
        clear_settings_cache()
        # Just verify it doesn't raise
        assert True

    def test_get_feature_flags(self, monkeypatch):
        """Test get_feature_flags function."""
        from src.infrastructure.config.settings import get_feature_flags, clear_settings_cache, FeatureFlags
        clear_settings_cache()
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "testing")

        flags = get_feature_flags()
        assert isinstance(flags, FeatureFlags)

    def test_is_feature_enabled(self, monkeypatch):
        """Test is_feature_enabled function."""
        from src.infrastructure.config.settings import is_feature_enabled, clear_settings_cache
        clear_settings_cache()
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "testing")

        assert is_feature_enabled("enable_audio_capture") is True


class TestEnvironmentPresets:
    """Tests for environment presets."""

    def test_development_preset(self, tmp_path, monkeypatch):
        """Test development environment preset."""
        from src.infrastructure.config.settings import load_settings, Environment, clear_settings_cache
        clear_settings_cache()
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "development")

        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        settings = load_settings(str(config_path), force_reload=True)
        assert settings.app.log_level == "DEBUG"
        assert settings.app.debug_mode is True
        assert settings.features.enable_debug_logging is True

    def test_production_preset(self, tmp_path, monkeypatch):
        """Test production environment preset."""
        from src.infrastructure.config.settings import load_settings, clear_settings_cache
        clear_settings_cache()
        monkeypatch.setenv("GAMECOACH_APP_ENVIRONMENT", "production")

        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}")

        settings = load_settings(str(config_path), force_reload=True)
        assert settings.app.log_level == "INFO"
        assert settings.app.debug_mode is False
