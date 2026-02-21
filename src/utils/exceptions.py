"""Re-exports of exceptions for backward compatibility.

This module re-exports all exception classes from src.exceptions
to maintain backward compatibility with existing imports.
"""

from src.exceptions import (
    # Base
    GameStudyError,
    # Vision
    VisionError,
    OCRError,
    DetectionError,
    ROIExtractionError,
    # Trigger
    TriggerError,
    TriggerConfigError,
    TriggerEvaluationError,
    # Dialogue
    DialogueError,
    APIError,
    RateLimitExceeded,
    RateLimitError,
    OpenAIError,
    TTSError,
    # Capture
    CaptureError,
    VideoCaptureError,
    ScreenCaptureError,
    # WebRTC
    WebRTCError,
    AuthenticationError,
    ConnectionError,
    # WebSocket
    WebSocketError,
    # Config
    ConfigError,
    ConfigurationError,
    FileNotFoundError,
    InvalidConfigError,
    # Model
    ModelLoadError,
    # Diagnostics
    DiagnosticsError,
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
