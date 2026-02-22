"""Session logger with JSONL support."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Cache for loggers
_loggers: dict[str, logging.Logger] = {}


class SensitiveFormatter(logging.Formatter):
    """Formatter that masks sensitive information like API keys and passwords."""

    # Patterns for sensitive data (key names and their values)
    SENSITIVE_PATTERNS = [
        (r'(api_key|api-key|apiKey|secret|token|password|credential|auth_key|auth-key)'
         r'[\'"]?\s*[:=]\s*[\'"]?([^\s\'",}]+)', r'\1=***REDACTED***'),
        (r'Bearer\s+([A-Za-z0-9\-._~+/]+)', r'Bearer ***REDACTED***'),
        (r'(sk-[a-zA-Z0-9]{20,})', r'sk-***REDACTED***'),
        (r'([A-Za-z0-9+/]{32,}={0,2})', r'***REDACTED***'),  # Base64-like strings
    ]

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        """Initialize formatter with optional format strings."""
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record and mask sensitive information.

        Args:
            record: Log record to format

        Returns:
            Formatted and sanitized log message
        """
        original = super().format(record)
        return self._mask_sensitive(original)

    def _mask_sensitive(self, text: str) -> str:
        """
        Mask sensitive information in text.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text with sensitive data masked
        """
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


def get_logger(name: str, use_sensitive_formatter: bool = True) -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name (usually __name__)
        use_sensitive_formatter: Whether to use SensitiveFormatter to mask sensitive data

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        if use_sensitive_formatter:
            handler.setFormatter(SensitiveFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    _loggers[name] = logger
    return logger


class SessionLogger:
    """Logger for game sessions with JSONL output."""

    def __init__(self, session_dir: str):
        """
        Initialize session logger.

        Args:
            session_dir: Directory for session logs
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.state_log_path = self.session_dir / "state.jsonl"
        self.trigger_log_path = self.session_dir / "triggers.jsonl"
        self.response_log_path = self.session_dir / "responses.jsonl"
        self.error_log_path = self.session_dir / "errors.log"

        # Also wrap logging methods with sensitive data filtering
        self._logger = get_logger(f"session.{id(self)}")

    def log_state(self, state: dict[str, Any]) -> None:
        """
        Log game state to JSONL file.

        Args:
            state: Game state dictionary
        """
        with open(self.state_log_path, 'a') as f:
            f.write(json.dumps(state, ensure_ascii=False) + '\n')

    def log_trigger(self, trigger: dict[str, Any]) -> None:
        """
        Log trigger event to JSONL file.

        Args:
            trigger: Trigger event dictionary
        """
        with open(self.trigger_log_path, 'a') as f:
            f.write(json.dumps(trigger, ensure_ascii=False) + '\n')

    def log_response(self, response: dict[str, Any]) -> None:
        """
        Log AI response to JSONL file.

        Args:
            response: Response dictionary
        """
        with open(self.response_log_path, 'a') as f:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')

    def log_error(self, message: str, exception: Exception | None = None) -> None:
        """
        Log error to error file.

        Args:
            message: Error message
            exception: Optional exception object
        """
        timestamp = datetime.now().isoformat()
        error_msg = f"[{timestamp}] {message}"
        if exception:
            error_msg += f"\n  Exception: {type(exception).__name__}: {exception}"

        with open(self.error_log_path, 'a') as f:
            f.write(error_msg + '\n')

    def get_session_info(self) -> dict[str, Any]:
        """
        Get session information.

        Returns:
            Session info dictionary
        """
        return {
            "session_dir": str(self.session_dir),
            "state_log_path": str(self.state_log_path),
            "trigger_log_path": str(self.trigger_log_path),
            "response_log_path": str(self.response_log_path),
            "created_at": datetime.now().isoformat(),
        }

    # Delegate standard logging methods to the internal logger
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log exception message."""
        self._logger.exception(msg, *args, **kwargs)
