"""Custom exception hierarchy for game-study project."""


class GameStudyError(Exception):
    """Base exception for all game-study errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize GameStudyError.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class VisionError(GameStudyError):
    """Error in vision/detection operations."""

    pass


class TriggerError(GameStudyError):
    """Error in trigger detection/processing."""

    pass


class DialogueError(GameStudyError):
    """Error in dialogue/AI operations."""

    pass


class CaptureError(GameStudyError):
    """Error in screen/video capture operations."""

    pass


class AudioError(GameStudyError):
    """Error in audio capture/processing operations."""

    pass


class APIError(GameStudyError):
    """Error in external API calls (OpenAI, etc.)."""

    pass


class ConfigurationError(GameStudyError):
    """Error in configuration/settings."""

    pass


class RateLimitError(APIError):
    """Error raised when API rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float = None):
        """
        Initialize RateLimitError.

        Args:
            message: Error message
            retry_after: Optional time in seconds to wait before retrying
        """
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after

        super().__init__(message, details)
        self.retry_after = retry_after
