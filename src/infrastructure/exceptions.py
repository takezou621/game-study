"""Infrastructure layer exceptions.

This module contains exception classes specific to the infrastructure layer.
All infrastructure exceptions inherit from InfrastructureError for consistent error handling.

Exception Hierarchy:
    InfrastructureError (base - inherits from GameStudyError)
    ├── ConnectionError - Connection failures
    ├── CaptureError - Video/screen capture failures
    ├── AudioError - Audio operation failures
    ├── DeviceNotFoundError - Device not found
    ├── LLMError - LLM operation failures
    ├── TTSError - TTS synthesis failures
    ├── OCRError - OCR processing failures
    ├── VisionError - Vision/detection failures
    └── ConfigurationLoadError - Configuration loading failures
"""

from typing import Any

from src.exceptions import GameStudyError


class InfrastructureError(GameStudyError):
    """Base exception for infrastructure layer errors.

    Infrastructure errors represent failures in external systems,
    I/O operations, and third-party integrations.

    Attributes:
        component: Infrastructure component that failed
        operation: Operation that was being performed
        retryable: Whether the operation can be retried
    """

    error_code: str = "INF000"

    def __init__(
        self,
        message: str,
        component: str | None = None,
        operation: str | None = None,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize InfrastructureError.

        Args:
            message: Human-readable error message
            component: Infrastructure component (e.g., "audio", "capture", "llm")
            operation: Operation that failed (e.g., "connect", "read", "write")
            retryable: Whether the operation can be retried
            context: Additional context information
            cause: Original exception that caused this error
        """
        context = context or {}
        context.update(
            {
                "component": component,
                "operation": operation,
                "retryable": retryable,
                "layer": "infrastructure",
            }
        )

        super().__init__(message, context=context, cause=cause)
        self.component = component
        self.operation = operation
        self.retryable = retryable


class ConnectionError(InfrastructureError):
    """Raised when a connection cannot be established or is lost.

    This covers WebSocket connections, API connections, and other
    network-based connections.

    Example:
        >>> raise ConnectionError(
        ...     service="OpenAI Realtime API",
        ...     reason="Connection timeout after 30 seconds",
        ...     retry_count=3
        ... )
    """

    error_code: str = "INF001"

    def __init__(
        self,
        service: str,
        reason: str,
        retry_count: int = 0,
        max_retries: int | None = None,
        retry_after_ms: int | None = None,
        **kwargs,
    ):
        """Initialize ConnectionError.

        Args:
            service: Service name that connection failed to
            reason: Reason for connection failure
            retry_count: Number of retry attempts made
            max_retries: Maximum number of retries allowed
            retry_after_ms: Suggested wait time before retry
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"Failed to connect to '{service}': {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "service": service,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "retry_after_ms": retry_after_ms,
            }
        )

        # Connection errors are often retryable
        kwargs.setdefault("retryable", True)
        kwargs.setdefault("component", "network")

        super().__init__(message, context=context, **kwargs)
        self.service = service
        self.reason = reason
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.retry_after_ms = retry_after_ms


class CaptureError(InfrastructureError):
    """Raised when video/screen capture fails.

    This covers capture from video files, screen capture, and webcam input.

    Example:
        >>> raise CaptureError(
        ...     source="screen://monitor:1",
        ...     reason="Permission denied for screen capture",
        ...     frame_number=1234
        ... )
    """

    error_code: str = "INF002"

    def __init__(
        self,
        source: str,
        reason: str,
        frame_number: int | None = None,
        timestamp_ms: int | None = None,
        **kwargs,
    ):
        """Initialize CaptureError.

        Args:
            source: Capture source identifier
            reason: Reason for failure
            frame_number: Frame number where error occurred (if applicable)
            timestamp_ms: Timestamp where error occurred (if applicable)
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"Capture failed for '{source}': {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "source": source,
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
            }
        )

        kwargs.setdefault("component", "capture")
        kwargs.setdefault("operation", "read")

        super().__init__(message, context=context, **kwargs)
        self.source = source
        self.reason = reason
        self.frame_number = frame_number
        self.timestamp_ms = timestamp_ms


class AudioError(InfrastructureError):
    """Raised when audio operations fail.

    This covers audio capture, playback, processing, and encoding/decoding.

    Example:
        >>> raise AudioError(
        ...     operation="capture",
        ...     reason="Audio device returned invalid sample rate",
        ...     device_index=0
        ... )
    """

    error_code: str = "INF003"

    def __init__(
        self,
        operation: str,
        reason: str,
        device_index: int | None = None,
        sample_rate: int | None = None,
        **kwargs,
    ):
        """Initialize AudioError.

        Args:
            operation: Audio operation (e.g., "capture", "playback", "encode")
            reason: Reason for failure
            device_index: Audio device index (if applicable)
            sample_rate: Sample rate being used (if applicable)
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"Audio operation '{operation}' failed: {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "device_index": device_index,
                "sample_rate": sample_rate,
            }
        )

        kwargs.setdefault("component", "audio")

        super().__init__(message, operation=operation, context=context, **kwargs)
        self.reason = reason
        self.device_index = device_index
        self.sample_rate = sample_rate


class DeviceNotFoundError(InfrastructureError):
    """Raised when an audio/video device is not found.

    Example:
        >>> raise DeviceNotFoundError(
        ...     device_type="audio",
        ...     device_id=2,
        ...     available_devices=[0, 1]
        ... )
    """

    error_code: str = "INF004"

    def __init__(
        self,
        device_type: str,
        device_id: str | int | None = None,
        available_devices: list[int] | None = None,
        **kwargs,
    ):
        """Initialize DeviceNotFoundError.

        Args:
            device_type: Type of device (e.g., "audio", "video", "capture")
            device_id: Device identifier that was not found
            available_devices: List of available device IDs
            **kwargs: Additional arguments passed to InfrastructureError
        """
        if device_id is not None:
            message = f"{device_type} device '{device_id}' not found"
        else:
            message = f"No {device_type} device available"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "device_type": device_type,
                "device_id": device_id,
                "available_devices": available_devices,
            }
        )

        kwargs.setdefault("component", device_type)
        kwargs.setdefault("retryable", False)

        super().__init__(message, context=context, **kwargs)
        self.device_type = device_type
        self.device_id = device_id
        self.available_devices = available_devices


class LLMError(InfrastructureError):
    """Raised when LLM operations fail.

    This covers failures in LLM API calls, response parsing, and
    authentication/authorization issues.

    Example:
        >>> raise LLMError(
        ...     provider="openai",
        ...     operation="chat_completion",
        ...     reason="Rate limit exceeded",
        ...     status_code=429
        ... )
    """

    error_code: str = "INF005"

    def __init__(
        self,
        provider: str,
        operation: str,
        reason: str,
        status_code: int | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """Initialize LLMError.

        Args:
            provider: LLM provider name (e.g., "openai", "anthropic")
            operation: Operation that failed (e.g., "chat", "embed", "stream")
            reason: Reason for failure
            status_code: HTTP status code (if applicable)
            model: Model being used (if applicable)
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"LLM '{provider}' operation '{operation}' failed: {reason}"
        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "provider": provider,
                "status_code": status_code,
                "model": model,
            }
        )

        # Rate limits and server errors are retryable
        if status_code in (429, 500, 502, 503, 504):
            kwargs.setdefault("retryable", True)

        kwargs.setdefault("component", "llm")

        super().__init__(message, operation=operation, context=context, **kwargs)
        self.provider = provider
        self.reason = reason
        self.status_code = status_code
        self.model = model


class TTSError(InfrastructureError):
    """Raised when TTS operations fail.

    Example:
        >>> raise TTSError(
        ...     reason="Voice 'custom' not available",
        ...     voice="custom",
        ...     provider="openai"
        ... )
    """

    error_code: str = "INF006"

    def __init__(
        self,
        reason: str,
        voice: str | None = None,
        provider: str | None = None,
        text_length: int | None = None,
        **kwargs,
    ):
        """Initialize TTSError.

        Args:
            reason: Reason for failure
            voice: Voice being used (if applicable)
            provider: TTS provider (if applicable)
            text_length: Length of text to synthesize (if applicable)
            **kwargs: Additional arguments passed to InfrastructureError
        """
        voice_info = f" (voice: {voice})" if voice else ""
        message = f"TTS synthesis failed{voice_info}: {reason}"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "voice": voice,
                "provider": provider,
                "text_length": text_length,
            }
        )

        kwargs.setdefault("component", "tts")

        super().__init__(message, context=context, **kwargs)
        self.reason = reason
        self.voice = voice
        self.provider = provider
        self.text_length = text_length


class OCRError(InfrastructureError):
    """Raised when OCR operations fail.

    Example:
        >>> raise OCRError(
        ...     reason="Failed to initialize Tesseract",
        ...     region="hp_bar",
        ...     engine="tesseract"
        ... )
    """

    error_code: str = "INF007"

    def __init__(
        self,
        reason: str,
        region: str | None = None,
        engine: str | None = None,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        """Initialize OCRError.

        Args:
            reason: Reason for failure
            region: Region being processed (if applicable)
            engine: OCR engine being used (if applicable)
            image_size: Size of image being processed (if applicable)
            **kwargs: Additional arguments passed to InfrastructureError
        """
        region_info = f" (region: {region})" if region else ""
        message = f"OCR failed{region_info}: {reason}"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "region": region,
                "engine": engine,
                "image_size": image_size,
            }
        )

        kwargs.setdefault("component", "ocr")

        super().__init__(message, context=context, **kwargs)
        self.reason = reason
        self.region = region
        self.engine = engine
        self.image_size = image_size


class VisionError(InfrastructureError):
    """Raised when vision/detection operations fail.

    This covers YOLO detection, feature extraction, and image processing.

    Example:
        >>> raise VisionError(
        ...     operation="yolo_detection",
        ...     reason="Model file not found: yolo11n.pt",
        ...     model="yolo11n"
        ... )
    """

    error_code: str = "INF008"

    def __init__(
        self,
        operation: str,
        reason: str,
        model: str | None = None,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        """Initialize VisionError.

        Args:
            operation: Vision operation (e.g., "detect", "classify", "extract")
            reason: Reason for failure
            model: Model being used (if applicable)
            image_size: Size of image being processed (if applicable)
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"Vision operation '{operation}' failed: {reason}"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "model": model,
                "image_size": image_size,
            }
        )

        kwargs.setdefault("component", "vision")

        super().__init__(message, operation=operation, context=context, **kwargs)
        self.reason = reason
        self.model = model
        self.image_size = image_size


class ConfigurationLoadError(InfrastructureError):
    """Raised when configuration cannot be loaded.

    Example:
        >>> raise ConfigurationLoadError(
        ...     config_path="./configs/app.yaml",
        ...     reason="File not found",
        ...     config_type="yaml"
        ... )
    """

    error_code: str = "INF009"

    def __init__(
        self,
        config_path: str,
        reason: str,
        config_type: str | None = None,
        **kwargs,
    ):
        """Initialize ConfigurationLoadError.

        Args:
            config_path: Path to configuration file
            reason: Reason for failure
            config_type: Type of configuration (e.g., "yaml", "json")
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"Failed to load configuration from '{config_path}': {reason}"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "config_path": config_path,
                "config_type": config_type,
            }
        )

        kwargs.setdefault("component", "config")
        kwargs.setdefault("operation", "load")

        super().__init__(message, context=context, **kwargs)
        self.config_path = config_path
        self.reason = reason
        self.config_type = config_type


class ResourceExhaustedError(InfrastructureError):
    """Raised when a resource is exhausted (memory, disk, rate limits).

    Example:
        >>> raise ResourceExhaustedError(
        ...     resource="memory",
        ...     reason="Frame buffer exceeded maximum size",
        ...     current_usage_mb=512,
        ...     limit_mb=256
        ... )
    """

    error_code: str = "INF010"

    def __init__(
        self,
        resource: str,
        reason: str,
        current_usage: Any = None,
        limit: Any = None,
        **kwargs,
    ):
        """Initialize ResourceExhaustedError.

        Args:
            resource: Resource that was exhausted (e.g., "memory", "disk", "rate_limit")
            reason: Reason for exhaustion
            current_usage: Current usage level
            limit: Resource limit
            **kwargs: Additional arguments passed to InfrastructureError
        """
        message = f"Resource '{resource}' exhausted: {reason}"

        context = kwargs.pop("context", None) or {}
        context.update(
            {
                "resource": resource,
                "current_usage": current_usage,
                "limit": limit,
            }
        )

        kwargs.setdefault("component", resource)
        kwargs.setdefault("retryable", resource == "rate_limit")

        super().__init__(message, context=context, **kwargs)
        self.resource = resource
        self.reason = reason
        self.current_usage = current_usage
        self.limit = limit


__all__ = [
    "InfrastructureError",
    "ConnectionError",
    "CaptureError",
    "AudioError",
    "DeviceNotFoundError",
    "LLMError",
    "TTSError",
    "OCRError",
    "VisionError",
    "ConfigurationLoadError",
    "ResourceExhaustedError",
]
