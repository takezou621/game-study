"""Presentation layer exceptions.

This module contains exception classes specific to the presentation layer.
All presentation exceptions inherit from PresentationError for consistent error handling.

Exception Hierarchy:
    PresentationError (base - inherits from GameStudyError)
    ├── CLIError - CLI-specific errors
    ├── OutputFormatError - Output formatting errors
    ├── UserInputError - User input validation errors
    └── DisplayError - Display/output errors
"""

from typing import Any

from src.exceptions import GameStudyError


class PresentationError(GameStudyError):
    """Base exception for presentation layer errors.

    Presentation errors represent failures in user interface, CLI,
    formatting, and output operations.

    Attributes:
        output_format: Output format being used (e.g., "text", "json")
        user_message: User-friendly message (safe to display)
    """

    error_code: str = "PRES000"

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        output_format: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize PresentationError.

        Args:
            message: Technical error message
            user_message: User-friendly message safe to display
            output_format: Output format being used
            context: Additional context information
            cause: Original exception that caused this error
        """
        context = context or {}
        context.update(
            {
                "output_format": output_format,
                "layer": "presentation",
            }
        )

        super().__init__(message, context=context, cause=cause)
        self.user_message = user_message or message
        self.output_format = output_format


class CLIError(PresentationError):
    """Raised when CLI operations fail.

    Example:
        >>> raise CLIError(
        ...     message="Invalid argument combination",
        ...     user_message="Cannot use --video with --input=screen",
        ...     exit_code=2
        ... )
    """

    error_code: str = "PRES001"

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        exit_code: int = 1,
        command: str | None = None,
        arguments: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize CLIError.

        Args:
            message: Technical error message
            user_message: User-friendly message
            exit_code: Suggested exit code for the CLI
            command: Command that failed
            arguments: Arguments that caused the issue
            **kwargs: Additional arguments passed to PresentationError
        """
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "exit_code": exit_code,
                "command": command,
                "arguments": arguments,
            }
        )

        super().__init__(message, user_message=user_message, context=context, **kwargs)
        self.exit_code = exit_code
        self.command = command
        self.arguments = arguments or {}


class OutputFormatError(PresentationError):
    """Raised when output formatting fails.

    Example:
        >>> raise OutputFormatError(
        ...     format="json",
        ...     reason="Object is not JSON serializable",
        ...     data_type="GameState"
        ... )
    """

    error_code: str = "PRES002"

    def __init__(
        self,
        format: str,
        reason: str,
        data_type: str | None = None,
        **kwargs,
    ):
        """Initialize OutputFormatError.

        Args:
            format: Output format that failed (e.g., "json", "text")
            reason: Reason for failure
            data_type: Type of data being formatted
            **kwargs: Additional arguments passed to PresentationError
        """
        message = f"Failed to format output as {format}: {reason}"
        user_message = "Error formatting output. Please try a different format."

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "format": format,
                "data_type": data_type,
            }
        )

        kwargs.setdefault("output_format", format)

        super().__init__(message, user_message=user_message, context=context, **kwargs)
        self.format = format
        self.reason = reason
        self.data_type = data_type


class UserInputError(PresentationError):
    """Raised when user input is invalid.

    Example:
        >>> raise UserInputError(
        ...     field="video_path",
        ...     value="/nonexistent.mp4",
        ...     reason="File does not exist"
        ... )
    """

    error_code: str = "PRES003"

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        suggestion: str | None = None,
        **kwargs,
    ):
        """Initialize UserInputError.

        Args:
            field: Input field name
            value: Invalid value provided
            reason: Reason why value is invalid
            suggestion: Suggested fix
            **kwargs: Additional arguments passed to PresentationError
        """
        message = f"Invalid input for '{field}': {reason}"
        user_message = f"Invalid {field}: {reason}"
        if suggestion:
            user_message += f" {suggestion}"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "field": field,
                "value": str(value) if value is not None else None,
                "suggestion": suggestion,
            }
        )

        super().__init__(message, user_message=user_message, context=context, **kwargs)
        self.field = field
        self.value = value
        self.reason = reason
        self.suggestion = suggestion


class DisplayError(PresentationError):
    """Raised when display/output operations fail.

    Example:
        >>> raise DisplayError(
        ...     operation="render",
        ...     reason="Terminal does not support colors"
        ... )
    """

    error_code: str = "PRES004"

    def __init__(
        self,
        operation: str,
        reason: str,
        **kwargs,
    ):
        """Initialize DisplayError.

        Args:
            operation: Display operation that failed
            reason: Reason for failure
            **kwargs: Additional arguments passed to PresentationError
        """
        message = f"Display operation '{operation}' failed: {reason}"
        user_message = "Error displaying output. Please check your terminal settings."

        context = kwargs.pop("context", None) or {}
        context["operation"] = operation

        super().__init__(message, user_message=user_message, context=context, **kwargs)
        self.operation = operation
        self.reason = reason


class FormatterError(PresentationError):
    """Raised when a formatter encounters an error.

    Example:
        >>> raise FormatterError(
        ...     formatter="JSONFormatter",
        ...     reason="Circular reference detected in object"
        ... )
    """

    error_code: str = "PRES005"

    def __init__(
        self,
        formatter: str,
        reason: str,
        **kwargs,
    ):
        """Initialize FormatterError.

        Args:
            formatter: Name of the formatter
            reason: Reason for failure
            **kwargs: Additional arguments passed to PresentationError
        """
        message = f"Formatter '{formatter}' error: {reason}"
        user_message = "Error formatting output."

        context = kwargs.pop("context", None) or {}
        context["formatter"] = formatter

        super().__init__(message, user_message=user_message, context=context, **kwargs)
        self.formatter = formatter
        self.reason = reason


__all__ = [
    "PresentationError",
    "CLIError",
    "OutputFormatError",
    "UserInputError",
    "DisplayError",
    "FormatterError",
]
