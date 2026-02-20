"""Custom exceptions for game-study application.

Provides a hierarchy of exceptions for proper error handling
and differentiation of error types.
"""


class GameStudyError(Exception):
    """Base exception for all game-study errors."""

    def __init__(self, message: str, cause: Exception = None):
        """
        Initialize exception.

        Args:
            message: Error message
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.cause = cause
        self.message = message

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# ============================================================================
# Vision Errors
# ============================================================================

class VisionError(GameStudyError):
    """Base exception for vision-related errors."""
    pass


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
    """Error communicating with external API."""

    def __init__(
        self,
        message: str,
        cause: Exception = None,
        retry_after: float = None
    ):
        """
        Initialize API error.

        Args:
            message: Error message
            cause: Original exception
            retry_after: Suggested wait time before retry (seconds)
        """
        super().__init__(message, cause)
        self.retry_after = retry_after


class RateLimitExceeded(APIError):
    """API rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        cause: Exception = None,
        retry_after: float = 60.0
    ):
        """
        Initialize rate limit error.

        Args:
            message: Error message
            cause: Original exception
            retry_after: Suggested wait time before retry (seconds)
        """
        super().__init__(message, cause, retry_after)


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
    """Base exception for capture-related errors."""
    pass


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
# Configuration Errors
# ============================================================================

class ConfigError(GameStudyError):
    """Error in configuration."""
    pass


class FileNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


class InvalidConfigError(ConfigError):
    """Invalid configuration value."""
    pass
