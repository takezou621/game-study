"""Application layer exceptions.

This module contains exception classes specific to the application layer.
All application exceptions inherit from ApplicationError for consistent error handling.

Exception Hierarchy:
    ApplicationError (base - inherits from GameStudyError)
    ├── UseCaseError - Use case execution failures
    ├── PortError - Port (interface) operation failures
    ├── ConfigurationError - Configuration issues
    ├── DTOParseError - DTO parsing failures
    └── ServiceError - Service operation failures
"""

from typing import Any

from src.exceptions import GameStudyError


class ApplicationError(GameStudyError):
    """Base exception for application layer errors.

    Application errors represent failures in application-level orchestration,
    use case execution, and coordination between domain and infrastructure.

    Attributes:
        use_case: Name of the use case that failed (if applicable)
        operation: Operation that was being performed
    """

    error_code: str = "APP000"

    def __init__(
        self,
        message: str,
        use_case: str | None = None,
        operation: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize ApplicationError.

        Args:
            message: Human-readable error message
            use_case: Name of the use case that failed
            operation: Operation that was being performed
            context: Additional context information
            cause: Original exception that caused this error
        """
        context = context or {}
        context.update(
            {
                "use_case": use_case,
                "operation": operation,
                "layer": "application",
            }
        )

        super().__init__(message, context=context, cause=cause)
        self.use_case = use_case
        self.operation = operation


class UseCaseError(ApplicationError):
    """Raised when a use case execution fails.

    This is the primary exception for use case failures, wrapping
    domain and infrastructure errors at the application layer.

    Example:
        >>> raise UseCaseError(
        ...     use_case="AnalyzeGameState",
        ...     reason="Failed to extract frame from capture source",
        ...     cause=original_exception
        ... )
    """

    error_code: str = "APP001"

    def __init__(
        self,
        use_case: str,
        reason: str,
        recoverable: bool = False,
        **kwargs,
    ):
        """Initialize UseCaseError.

        Args:
            use_case: Name of the use case
            reason: Reason for failure
            recoverable: Whether the error is potentially recoverable
            **kwargs: Additional arguments passed to ApplicationError
        """
        message = f"Use case '{use_case}' failed: {reason}"
        context = kwargs.pop("context", None) or {}
        context["recoverable"] = recoverable

        super().__init__(message, use_case=use_case, context=context, **kwargs)
        self.reason = reason
        self.recoverable = recoverable


class PortError(ApplicationError):
    """Raised when a port (interface) operation fails.

    Port errors indicate failures in the adapter layer when implementing
    port interfaces (e.g., capture, LLM, TTS ports).

    Example:
        >>> raise PortError(
        ...     port_name="CapturePort",
        ...     operation="read",
        ...     reason="Frame capture timeout",
        ...     adapter="VideoCaptureAdapter"
        ... )
    """

    error_code: str = "APP002"

    def __init__(
        self,
        port_name: str,
        operation: str,
        reason: str,
        adapter: str | None = None,
        **kwargs,
    ):
        """Initialize PortError.

        Args:
            port_name: Name of the port interface
            operation: Operation that failed (e.g., "read", "connect")
            reason: Reason for failure
            adapter: Name of the adapter implementation
            **kwargs: Additional arguments passed to ApplicationError
        """
        message = f"Port '{port_name}' operation '{operation}' failed: {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "port_name": port_name,
                "operation": operation,
                "adapter": adapter,
            }
        )

        super().__init__(message, operation=operation, context=context, **kwargs)
        self.port_name = port_name
        self.operation = operation
        self.reason = reason
        self.adapter = adapter


class ConfigurationError(ApplicationError):
    """Raised when configuration is invalid or missing.

    Configuration errors indicate problems with application configuration,
    such as missing required settings or invalid values.

    Example:
        >>> raise ConfigurationError(
        ...     config_key="llm.api_key",
        ...     reason="API key is required but not provided",
        ...     suggestion="Set OPENAI_API_KEY environment variable"
        ... )
    """

    error_code: str = "APP003"

    def __init__(
        self,
        config_key: str,
        reason: str,
        config_value: Any = None,
        suggestion: str | None = None,
        **kwargs,
    ):
        """Initialize ConfigurationError.

        Args:
            config_key: Configuration key with issue
            reason: Reason for failure
            config_value: Current value (if any, may be redacted)
            suggestion: Suggested fix
            **kwargs: Additional arguments passed to ApplicationError
        """
        message = f"Configuration error for '{config_key}': {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "config_key": config_key,
                "suggestion": suggestion,
            }
        )

        super().__init__(message, context=context, **kwargs)
        self.config_key = config_key
        self.reason = reason
        self.config_value = config_value
        self.suggestion = suggestion


class DTOParseError(ApplicationError):
    """Raised when DTO parsing fails.

    DTO parse errors indicate problems converting data between
    different layers or formats.

    Example:
        >>> raise DTOParseError(
        ...     dto_type="FrameDTO",
        ...     errors=[{"field": "timestamp", "error": "Invalid format"}]
        ... )
    """

    error_code: str = "APP004"

    def __init__(
        self,
        dto_type: str,
        errors: list[dict[str, Any]],
        raw_data: Any = None,
        **kwargs,
    ):
        """Initialize DTOParseError.

        Args:
            dto_type: Type of DTO being parsed
            errors: List of parsing errors with field and message
            raw_data: Original data that failed to parse (may be truncated)
            **kwargs: Additional arguments passed to ApplicationError
        """
        message = f"Failed to parse {dto_type}: {len(errors)} error(s)"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "dto_type": dto_type,
                "errors": errors,
                "error_count": len(errors),
            }
        )

        super().__init__(message, context=context, **kwargs)
        self.dto_type = dto_type
        self.errors = errors
        self.raw_data = raw_data


class ServiceError(ApplicationError):
    """Raised when a service operation fails.

    Service errors indicate failures in application services that
    coordinate between use cases and domain/infrastructure.

    Example:
        >>> raise ServiceError(
        ...     service_name="GameAnalyzerService",
        ...     operation="analyze",
        ...     reason="State validation failed",
        ...     cause=validation_exception
        ... )
    """

    error_code: str = "APP005"

    def __init__(
        self,
        service_name: str,
        operation: str,
        reason: str,
        **kwargs,
    ):
        """Initialize ServiceError.

        Args:
            service_name: Name of the service
            operation: Operation that failed
            reason: Reason for failure
            **kwargs: Additional arguments passed to ApplicationError
        """
        message = f"Service '{service_name}' operation '{operation}' failed: {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "service_name": service_name,
                "operation": operation,
            }
        )

        super().__init__(message, operation=operation, context=context, **kwargs)
        self.service_name = service_name
        self.operation = operation
        self.reason = reason


class OrchestrationError(ApplicationError):
    """Raised when orchestration between components fails.

    This is for errors that occur when coordinating multiple
    components or services together.

    Example:
        >>> raise OrchestrationError(
        ...     workflow="session_lifecycle",
        ...     stage="initialization",
        ...     reason="Failed to initialize capture and audio simultaneously"
        ... )
    """

    error_code: str = "APP006"

    def __init__(
        self,
        workflow: str,
        stage: str,
        reason: str,
        failed_components: list[str] | None = None,
        **kwargs,
    ):
        """Initialize OrchestrationError.

        Args:
            workflow: Name of the workflow
            stage: Stage where failure occurred
            reason: Reason for failure
            failed_components: List of components that failed
            **kwargs: Additional arguments passed to ApplicationError
        """
        message = f"Orchestration failed in '{workflow}' at stage '{stage}': {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "workflow": workflow,
                "stage": stage,
                "failed_components": failed_components,
            }
        )

        super().__init__(message, context=context, **kwargs)
        self.workflow = workflow
        self.stage = stage
        self.reason = reason
        self.failed_components = failed_components or []


__all__ = [
    "ApplicationError",
    "UseCaseError",
    "PortError",
    "ConfigurationError",
    "DTOParseError",
    "ServiceError",
    "OrchestrationError",
]
