"""Custom exceptions for game-study application.

Provides a hierarchy of exceptions for proper error handling
and differentiation of error types.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional


# ============================================================================
# Base Exception
# ============================================================================

class GameStudyError(Exception):
    """Base exception for all game-study errors.

    Attributes:
        message: Human-readable error message
        context: Dictionary with additional error context
        cause: Original exception that caused this error
        timestamp: When the error occurred
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize GameStudyError.

        Args:
            message: Human-readable error message
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        """Return string representation with context."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[{context_str}]")

        if self.cause:
            parts.append(f"caused by: {type(self.cause).__name__}: {self.cause}")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "cause": type(self.cause).__name__ if self.cause else None,
            "timestamp": self.timestamp.isoformat(),
        }

    def log(self, level: int = logging.ERROR) -> None:
        """Log this exception with context.

        Args:
            level: Logging level (default: ERROR)
        """
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.log(level, self.message, extra={
            "exception_type": self.__class__.__name__,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        })


# ============================================================================
# Vision Errors
# ============================================================================

class VisionError(GameStudyError):
    """Base exception for vision-related errors.

    This includes OCR failures, YOLO detection errors,
    image processing problems, etc.

    Attributes:
        detection_type: Type of detection that failed (e.g., "ocr", "yolo")
        frame_shape: Shape of the frame being processed
    """

    def __init__(
        self,
        message: str,
        detection_type: Optional[str] = None,
        frame_shape: Optional[tuple] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize VisionError.

        Args:
            message: Human-readable error message
            detection_type: Type of detection that failed
            frame_shape: Shape of the frame
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({
            "detection_type": detection_type,
            "frame_shape": frame_shape,
        })

        super().__init__(message, context=context, cause=cause)
        self.detection_type = detection_type
        self.frame_shape = frame_shape


class OCRError(VisionError):
    """Error during OCR processing."""
    pass


class DetectionError(VisionError):
    """Error during object detection."""
    pass


class ROIExtractionError(VisionError):
    """Error extracting ROI from frame."""
    pass


# ============================================================================
# Trigger Errors
# ============================================================================

class TriggerError(GameStudyError):
    """Base exception for trigger-related errors."""
    pass


class TriggerConfigError(TriggerError):
    """Error in trigger configuration."""
    pass


class TriggerEvaluationError(TriggerError):
    """Error during trigger evaluation."""
    pass


# ============================================================================
# Dialogue Errors
# ============================================================================

class DialogueError(GameStudyError):
    """Base exception for dialogue-related errors."""
    pass


class APIError(DialogueError):
    """Error communicating with external API.

    Attributes:
        status_code: HTTP status code if applicable
        endpoint: API endpoint that was called
        retryable: Whether this error is retryable
        retry_after: Suggested wait time before retry (seconds)
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            endpoint: API endpoint that failed
            retryable: Whether this error can be retried
            retry_after: Suggested wait time before retry (seconds)
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({
            "status_code": status_code,
            "endpoint": endpoint,
            "retryable": retryable,
        })

        super().__init__(message, context=context, cause=cause)
        self.status_code = status_code
        self.endpoint = endpoint
        self.retryable = retryable
        self.retry_after = retry_after

    def is_retryable(self) -> bool:
        """Check if this error is retryable."""
        return self.retryable

    @classmethod
    def from_response(
        cls,
        endpoint: str,
        status_code: int,
        response_text: str
    ) -> "APIError":
        """Create APIError from HTTP response.

        Args:
            endpoint: API endpoint
            status_code: HTTP status code
            response_text: Response body

        Returns:
            APIError instance
        """
        retryable = status_code in {408, 429, 500, 502, 503, 504}

        return cls(
            message=f"API request to {endpoint} failed with status {status_code}",
            status_code=status_code,
            endpoint=endpoint,
            retryable=retryable,
            context={"response_text": response_text[:500]}  # Truncate long responses
        )


class RateLimitExceeded(APIError):
    """API rate limit exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying
        limit: Rate limit that was exceeded
        remaining: Remaining requests (if available)
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        cause: Optional[Exception] = None,
        retry_after: float = 60.0,
        limit: Optional[str] = None,
        remaining: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            cause: Original exception
            retry_after: Suggested wait time before retry (seconds)
            limit: Rate limit description
            remaining: Remaining requests
            context: Additional context
        """
        context = context or {}
        context.update({
            "retry_after": retry_after,
            "limit": limit,
            "remaining": remaining,
        })

        super().__init__(
            message,
            status_code=429,
            retryable=True,
            retry_after=retry_after,
            context=context,
            cause=cause
        )
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


# Alias for compatibility
RateLimitError = RateLimitExceeded


class OpenAIError(APIError):
    """Error from OpenAI API."""
    pass


class TTSError(DialogueError):
    """Error during text-to-speech synthesis."""
    pass


# ============================================================================
# Capture Errors
# ============================================================================

class CaptureError(GameStudyError):
    """Base exception for capture-related errors.

    This includes display connection issues, permission errors,
    and capture device problems.

    Attributes:
        capture_type: Type of capture (e.g., "screen", "window", "webcam")
        monitor: Monitor number or identifier
    """

    def __init__(
        self,
        message: str,
        capture_type: Optional[str] = None,
        monitor: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize CaptureError.

        Args:
            message: Human-readable error message
            capture_type: Type of capture that failed
            monitor: Monitor identifier
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({
            "capture_type": capture_type,
            "monitor": monitor,
        })

        super().__init__(message, context=context, cause=cause)
        self.capture_type = capture_type
        self.monitor = monitor


class VideoCaptureError(CaptureError):
    """Error capturing video."""
    pass


class ScreenCaptureError(CaptureError):
    """Error capturing screen."""
    pass


# ============================================================================
# WebRTC Errors
# ============================================================================

class WebRTCError(GameStudyError):
    """Base exception for WebRTC-related errors."""
    pass


class AuthenticationError(WebRTCError):
    """Authentication failed."""
    pass


class ConnectionError(WebRTCError):
    """WebRTC connection error."""
    pass


# ============================================================================
# WebSocket Errors
# ============================================================================

class WebSocketError(GameStudyError):
    """Exception raised for WebSocket connection errors.

    This includes connection failures, disconnection during use,
    and message send/receive errors.

    Attributes:
        url: WebSocket URL
        state: Connection state when error occurred
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        state: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize WebSocketError.

        Args:
            message: Human-readable error message
            url: WebSocket URL
            state: Connection state
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({
            "url": url,
            "state": state,
        })

        super().__init__(message, context=context, cause=cause)
        self.url = url
        self.state = state


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigError(GameStudyError):
    """Error in configuration.

    Attributes:
        config_key: Configuration key that caused the error
        config_file: Configuration file path (if applicable)
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize ConfigError.

        Args:
            message: Human-readable error message
            config_key: Configuration key
            config_file: Configuration file path
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({
            "config_key": config_key,
            "config_file": config_file,
        })

        super().__init__(message, context=context, cause=cause)
        self.config_key = config_key
        self.config_file = config_file


# Alias for compatibility
ConfigurationError = ConfigError


class FileNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


class InvalidConfigError(ConfigError):
    """Invalid configuration value."""
    pass


# ============================================================================
# Model Errors
# ============================================================================

class ModelLoadError(GameStudyError):
    """Exception raised when model loading fails.

    This includes missing model files, incompatible formats,
    and initialization errors.

    Attributes:
        model_path: Path to the model file
        model_type: Type of model (e.g., "yolo", "tesseract")
    """

    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize ModelLoadError.

        Args:
            message: Human-readable error message
            model_path: Path to the model
            model_type: Type of model
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({
            "model_path": model_path,
            "model_type": model_type,
        })

        super().__init__(message, context=context, cause=cause)
        self.model_path = model_path
        self.model_type = model_type


# ============================================================================
# Diagnostics Errors
# ============================================================================

class DiagnosticsError(GameStudyError):
    """Base exception for diagnostics errors.

    This includes audio diagnostics, system checks, and
    report generation failures.

    Attributes:
        check_type: Type of diagnostic check that failed
    """

    def __init__(
        self,
        message: str,
        check_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize DiagnosticsError.

        Args:
            message: Human-readable error message
            check_type: Type of diagnostic check
            context: Additional context
            cause: Original exception
        """
        context = context or {}
        context.update({"check_type": check_type})

        super().__init__(message, context=context, cause=cause)
        self.check_type = check_type


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
