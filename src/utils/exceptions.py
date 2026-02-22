"""Re-exports of exceptions for backward compatibility.

This module re-exports all exception classes from src.exceptions
to maintain backward compatibility with existing imports.
"""

from src.exceptions import (
    APIError,
    AuthenticationError,
    # Capture
    CaptureError,
    # Config
    ConfigError,
    ConfigurationError,
    ConnectionError,
    DetectionError,
    # Diagnostics
    DiagnosticsError,
    # Dialogue
    DialogueError,
    FileNotFoundError,
    # Base
    GameStudyError,
    InvalidConfigError,
    # Model
    ModelLoadError,
    OCRError,
    OpenAIError,
    RateLimitError,
    RateLimitExceeded,
    ROIExtractionError,
    ScreenCaptureError,
    TriggerConfigError,
    # Trigger
    TriggerError,
    TriggerEvaluationError,
    TTSError,
    VideoCaptureError,
    # Vision
    VisionError,
    # WebRTC
    WebRTCError,
    # WebSocket
    WebSocketError,
)

__all__ = [
    # Base
    "GameStudyError",
    # Vision
    "VisionError",
    "OCRError",
    "DetectionError",
    "ROIExtractionError",
    # Trigger
    "TriggerError",
    "TriggerConfigError",
    "TriggerEvaluationError",
    # Dialogue
    "DialogueError",
    "APIError",
    "RateLimitExceeded",
    "RateLimitError",
    "OpenAIError",
    "TTSError",
    # Capture
    "CaptureError",
    "VideoCaptureError",
    "ScreenCaptureError",
    # WebRTC
    "WebRTCError",
    "AuthenticationError",
    "ConnectionError",
    # WebSocket
    "WebSocketError",
    # Config
    "ConfigError",
    "ConfigurationError",
    "FileNotFoundError",
    "InvalidConfigError",
    # Model
    "ModelLoadError",
    # Diagnostics
    "DiagnosticsError",
]
