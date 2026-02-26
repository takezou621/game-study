"""Application settings with YAML configuration, environment variables, and Pydantic validation."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from utils.logger import get_logger

logger = get_logger(__name__)

# Environment prefix for all environment variables
ENV_PREFIX = "GAMECOACH_"


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class FeatureFlags(BaseModel):
    """Feature flags for enabling/disabling application features."""

    model_config = ConfigDict(extra="forbid")

    enable_audio_capture: bool = Field(
        default=True,
        description="Enable audio capture functionality",
    )
    enable_screen_capture: bool = Field(
        default=True,
        description="Enable screen capture functionality",
    )
    enable_video_capture: bool = Field(
        default=True,
        description="Enable video file capture functionality",
    )
    enable_realtime_api: bool = Field(
        default=True,
        description="Enable OpenAI Realtime API",
    )
    enable_vad: bool = Field(
        default=True,
        description="Enable Voice Activity Detection",
    )
    enable_noise_gate: bool = Field(
        default=True,
        description="Enable noise gate processing",
    )
    enable_debug_logging: bool = Field(
        default=False,
        description="Enable detailed debug logging",
    )
    enable_performance_metrics: bool = Field(
        default=False,
        description="Enable performance metrics collection",
    )


class AppSettings(BaseModel):
    """Application-level settings."""

    model_config = ConfigDict(extra="forbid")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    output_dir: str = Field(
        default="./logs",
        description="Directory for log output",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Validate and normalize output directory path."""
        # Expand user home directory if present
        expanded = os.path.expanduser(v)
        return expanded


class AudioSettings(BaseModel):
    """Audio configuration settings."""

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Audio sample rate in Hz",
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Audio chunk size in samples",
    )
    noise_gate_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Noise gate threshold level",
    )
    noise_gate_attack_ms: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Noise gate attack time in milliseconds",
    )
    noise_gate_release_ms: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description="Noise gate release time in milliseconds",
    )
    vad_enabled: bool = Field(
        default=True,
        description="Enable Voice Activity Detection",
    )
    vad_padding_ms: int = Field(
        default=300,
        ge=0,
        le=1000,
        description="VAD padding in milliseconds",
    )
    vad_min_speech_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Minimum speech duration in milliseconds",
    )
    vad_max_speech_ms: int = Field(
        default=10000,
        ge=1000,
        le=60000,
        description="Maximum speech duration in milliseconds",
    )
    device_index: int | None = Field(
        default=None,
        ge=0,
        description="Audio device index (None for default)",
    )

    @model_validator(mode="after")
    def validate_vad_timing(self) -> AudioSettings:
        """Validate VAD timing relationships."""
        if self.vad_min_speech_ms >= self.vad_max_speech_ms:
            raise ValueError(
                f"vad_min_speech_ms ({self.vad_min_speech_ms}) must be less than "
                f"vad_max_speech_ms ({self.vad_max_speech_ms})"
            )
        return self


class CaptureSettings(BaseModel):
    """Capture configuration settings."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["video", "screen"] = Field(
        default="video",
        description="Capture type: 'video' or 'screen'",
    )
    video_path: str | None = Field(
        default=None,
        description="Path to video file (for video capture type)",
    )
    loop_video: bool = Field(
        default=False,
        description="Loop video playback",
    )
    monitor_index: int = Field(
        default=1,
        ge=0,
        description="Monitor index for screen capture",
    )
    region: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Capture region as (x, y, width, height)",
    )

    @field_validator("region", mode="before")
    @classmethod
    def validate_region(cls, v: Any) -> tuple[int, int, int, int] | None:
        """Validate and convert region to tuple."""
        if v is None:
            return None
        if isinstance(v, list | tuple) and len(v) == 4:
            region = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
            if any(x < 0 for x in region):
                raise ValueError("Region values must be non-negative")
            return region
        raise ValueError("Region must be a list or tuple of 4 integers")

    @model_validator(mode="after")
    def validate_video_path_for_video_type(self) -> CaptureSettings:
        """Validate video_path is set when type is 'video'."""
        if self.type == "video" and self.video_path is not None:
            # Only warn, don't fail - video might be set later
            path = Path(self.video_path)
            if not path.exists() and not self.video_path.startswith(("http://", "https://")):
                logger.warning(f"Video path does not exist: {self.video_path}")
        return self


class TriggerSettings(BaseModel):
    """Trigger configuration settings."""

    model_config = ConfigDict(extra="forbid")

    config_path: str = Field(
        default="./configs/triggers.yaml",
        description="Path to triggers configuration file",
    )
    cooldown_ms: int = Field(
        default=5000,
        ge=0,
        le=60000,
        description="Cooldown period between triggers in milliseconds",
    )
    max_response_length_ms: int = Field(
        default=10000,
        ge=1000,
        le=120000,
        description="Maximum response length in milliseconds",
    )


class LLMSettings(BaseModel):
    """LLM configuration settings."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (use environment variable for security)",
    )
    model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use",
    )
    voice: str = Field(
        default="alloy",
        description="Voice for audio responses",
    )
    use_realtime_api: bool = Field(
        default=True,
        description="Use OpenAI Realtime API",
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=4096,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Response temperature (randomness)",
    )
    system_prompt_path: str = Field(
        default="./configs/prompts/system.txt",
        description="Path to system prompt file",
    )

    def get_api_key(self) -> str | None:
        """Get the API key value, returning None if not set."""
        if self.api_key is None:
            return None
        return self.api_key.get_secret_value()

    def model_dump_safe(self) -> dict[str, Any]:
        """Dump model with sensitive fields masked."""
        data = self.model_dump()
        if "api_key" in data and data["api_key"] is not None:
            data["api_key"] = "***REDACTED***"
        return data


class Settings(BaseModel):
    """Complete application settings with environment variable support."""

    model_config = ConfigDict(extra="forbid")

    app: AppSettings = Field(default_factory=AppSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    trigger: TriggerSettings = Field(default_factory=TriggerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    def model_dump_safe(self) -> dict[str, Any]:
        """Dump settings with sensitive fields masked."""
        data = self.model_dump()
        if "llm" in data:
            data["llm"] = self.llm.model_dump_safe()
        return data


# Preset configurations for different environments
ENVIRONMENT_PRESETS: dict[Environment, dict[str, Any]] = {
    Environment.DEVELOPMENT: {
        "app": {
            "log_level": "DEBUG",
            "debug_mode": True,
            "environment": Environment.DEVELOPMENT,
        },
        "features": {
            "enable_debug_logging": True,
            "enable_performance_metrics": True,
        },
        "llm": {
            "temperature": 0.8,
        },
    },
    Environment.TESTING: {
        "app": {
            "log_level": "WARNING",
            "debug_mode": False,
            "environment": Environment.TESTING,
        },
        "features": {
            "enable_debug_logging": False,
            "enable_performance_metrics": False,
        },
        "llm": {
            "temperature": 0.0,
            "max_tokens": 100,
        },
    },
    Environment.PRODUCTION: {
        "app": {
            "log_level": "INFO",
            "debug_mode": False,
            "environment": Environment.PRODUCTION,
        },
        "features": {
            "enable_debug_logging": False,
            "enable_performance_metrics": True,
        },
        "llm": {
            "temperature": 0.7,
        },
    },
}


def _get_env_var(section: str, key: str) -> str | None:
    """
    Get environment variable value using the standard prefix pattern.

    Format: GAMECOACH_{SECTION}_{KEY} (e.g., GAMECOACH_AUDIO_SAMPLE_RATE)
    """
    env_key = f"{ENV_PREFIX}{section.upper()}_{key.upper()}"
    return os.environ.get(env_key)


def _parse_env_value(value: str, target_type: type) -> Any:
    """Parse environment variable string to the target type."""
    if target_type is bool:
        return value.lower() in ("true", "1", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _apply_env_variables(data: dict[str, Any], settings_model: type[BaseModel]) -> dict[str, Any]:
    """
    Apply environment variables to configuration data.

    This function overlays environment variables onto the configuration data,
    with environment variables taking precedence over file-based configuration.
    """
    result = dict(data)

    # Get field info from the model
    for field_name, field_info in settings_model.model_fields.items():
        field_type = field_info.annotation
        field_name_upper = field_name.upper()

        # Handle nested models (sections like 'app', 'audio', etc.)
        if (
            field_type is not None
            and isinstance(field_type, type)
            and issubclass(field_type, BaseModel)
        ):
            if field_name not in result:
                result[field_name] = {}
            if not isinstance(result[field_name], dict):
                result[field_name] = result[field_name].model_dump()

            # Process nested fields
            for nested_field_name, nested_field_info in field_type.model_fields.items():
                env_value = _get_env_var(field_name, nested_field_name)
                if env_value is not None:
                    nested_type = nested_field_info.annotation
                    # Handle Optional types - default to str if we can't determine type
                    actual_type: type = str
                    if nested_type is not None:
                        if hasattr(nested_type, "__origin__"):
                            origin = getattr(nested_type, "__origin__", None)
                            if origin is not None and origin is type(None) | type:
                                args = getattr(nested_type, "__args__", ())
                                if args:
                                    actual_type = args[0]
                            else:
                                actual_type = nested_type
                        else:
                            actual_type = nested_type
                    try:
                        result[field_name][nested_field_name] = _parse_env_value(
                            env_value, actual_type
                        )
                        logger.debug(
                            f"Applied env var GAMECOACH_{field_name_upper}_{nested_field_name.upper()}"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to parse env var GAMECOACH_{field_name_upper}_{nested_field_name.upper()}: {e}"
                        )

        # Handle top-level fields (like 'features' at root level)
        else:
            env_value = _get_env_var("", field_name)
            if env_value is not None and field_type is not None:
                field_actual_type: type = str
                if hasattr(field_type, "__origin__"):
                    origin = getattr(field_type, "__origin__", None)
                    if origin is not None and origin is type(None) | type:
                        args = getattr(field_type, "__args__", ())
                        if args:
                            field_actual_type = args[0]
                    else:
                        field_actual_type = field_type
                else:
                    field_actual_type = field_type
                try:
                    result[field_name] = _parse_env_value(env_value, field_actual_type)
                    logger.debug(f"Applied env var GAMECOACH_{field_name_upper}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse env var GAMECOACH_{field_name_upper}: {e}")

    return result


def _apply_preset(data: dict[str, Any], environment: Environment) -> dict[str, Any]:
    """Apply environment preset to configuration data."""
    preset = ENVIRONMENT_PRESETS.get(environment, {})
    result = dict(data)

    for section, values in preset.items():
        if section not in result:
            result[section] = {}
        if isinstance(result[section], dict) and isinstance(values, dict):
            # Preset values take lower priority than explicit config
            merged = dict(values)
            merged.update(result[section])
            result[section] = merged
        else:
            result[section] = values

    return result


def _load_secrets_from_env(data: dict[str, Any]) -> dict[str, Any]:
    """Load sensitive values from specific environment variables."""
    result = dict(data)

    # Handle OpenAI API key from standard environment variable
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        if "llm" not in result:
            result["llm"] = {}
        result["llm"]["api_key"] = openai_key
        logger.debug("Loaded API key from OPENAI_API_KEY environment variable")

    # Also support prefixed version
    prefixed_key = os.environ.get(f"{ENV_PREFIX}LLM_API_KEY")
    if prefixed_key:
        if "llm" not in result:
            result["llm"] = {}
        result["llm"]["api_key"] = prefixed_key
        logger.debug("Loaded API key from GAMECOACH_LLM_API_KEY environment variable")

    return result


# Cache for settings (stores the last loaded settings)
_settings_cache: Settings | None = None


def load_settings(
    path: str | None = None,
    environment: Environment | None = None,
    apply_preset: bool = True,
    apply_env: bool = True,
    force_reload: bool = False,
) -> Settings:
    """
    Load settings from YAML file with environment variable and preset support.

    Settings are loaded in the following order (later takes precedence):
    1. Environment preset (based on environment parameter or GAMECOACH_APP_ENVIRONMENT)
    2. YAML configuration file
    3. Environment variables (GAMECOACH_* pattern)
    4. Secrets from environment (OPENAI_API_KEY, etc.)

    Args:
        path: Path to settings file. If None, uses default locations.
        environment: Environment to use for presets. If None, reads from
                     GAMECOACH_APP_ENVIRONMENT or defaults to DEVELOPMENT.
        apply_preset: Whether to apply environment preset defaults.
        apply_env: Whether to apply environment variable overrides.
        force_reload: If True, bypass cache and reload settings.

    Returns:
        Settings object with all configurations merged.
    """
    default_paths = [
        Path("./configs/app.yaml"),
        Path("./config.yaml"),
        Path("./settings.yaml"),
    ]

    # Determine environment
    if environment is None:
        env_str = os.environ.get(
            f"{ENV_PREFIX}APP_ENVIRONMENT",
            os.environ.get("ENVIRONMENT", "development"),
        )
        try:
            environment = Environment(env_str.lower())
        except ValueError:
            logger.warning(f"Invalid environment '{env_str}', defaulting to development")
            environment = Environment.DEVELOPMENT

    # Load from file
    data: dict[str, Any] = {}
    config_paths = [Path(path)] if path else default_paths

    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"Loading settings from {config_path}")
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            break
    else:
        logger.info("No config file found, using defaults")

    # Apply preset defaults first
    if apply_preset and environment:
        data = _apply_preset(data, environment)

    # Load secrets from environment
    data = _load_secrets_from_env(data)

    # Apply environment variable overrides
    if apply_env:
        data = _apply_env_variables(data, Settings)

    # Use cache if available and not forcing reload
    global _settings_cache
    if _settings_cache is not None and not force_reload:
        return _settings_cache

    # Create and validate settings
    settings = Settings.model_validate(data)

    # Cache the settings
    _settings_cache = settings

    logger.info(f"Settings loaded for environment: {settings.app.environment.value}")

    return settings


def save_settings(settings: Settings, path: str, include_secrets: bool = False) -> None:
    """
    Save settings to YAML file.

    Args:
        settings: Settings to save.
        path: Path to save to.
        include_secrets: Whether to include sensitive values (not recommended).
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if include_secrets:
        data = settings.model_dump()
    else:
        data = settings.model_dump_safe()

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    logger.info(f"Settings saved to {path}")


def clear_settings_cache() -> None:
    """Clear the settings cache, forcing reload on next access."""
    global _settings_cache
    _settings_cache = None


def get_feature_flags() -> FeatureFlags:
    """Get the current feature flags."""
    return load_settings().features


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a feature is enabled.

    Args:
        feature_name: Name of the feature flag to check.

    Returns:
        True if the feature is enabled, False otherwise.

    Raises:
        AttributeError: If the feature name doesn't exist.
    """
    flags = get_feature_flags()
    value = getattr(flags, feature_name)
    return bool(value)
