"""Application layer for game coach.

This layer contains:
- Use Cases: Application-specific business rules
- Services: Application services
- DTOs: Data Transfer Objects
- Ports: Interfaces for external dependencies
- Exceptions: Application-specific exceptions
"""

from application.exceptions import (
    ApplicationError,
    ConfigurationError,
    DTOParseError,
    OrchestrationError,
    PortError,
    ServiceError,
    UseCaseError,
)
from application.ports.audio_port import AudioPort
from application.ports.capture_port import CapturePort
from application.ports.llm_port import LLMPort
from application.ports.tts_port import TTSPort

__all__ = [
    # Ports
    "CapturePort",
    "LLMPort",
    "TTSPort",
    "AudioPort",
    # Exceptions
    "ApplicationError",
    "UseCaseError",
    "PortError",
    "ConfigurationError",
    "DTOParseError",
    "ServiceError",
    "OrchestrationError",
]
