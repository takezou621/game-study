"""
Optional dependency management utilities.

This module provides helpers for graceful handling of optional dependencies
throughout the codebase. Instead of repeating try-except ImportError patterns,
use these utilities for consistent dependency checking.
"""

from importlib import import_module
from typing import Any, TypeVar

T = TypeVar("T")


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is required but not available."""

    def __init__(self, package_name: str, feature: str = "") -> None:
        message = f"Missing optional dependency: {package_name}"
        if feature:
            message += f" (required for {feature})"
        message += f". Install with: pip install {package_name}"
        super().__init__(message)


def optional_import(
    module_name: str,
    package_name: str | None = None,
    attribute: str | None = None,
) -> tuple[bool, Any]:
    """
    Safely import an optional module or attribute.

    Args:
        module_name: The module to import (e.g., 'openai', 'numpy')
        package_name: The pip package name for install hints (defaults to module_name)
        attribute: Optional attribute to extract from the module

    Returns:
        Tuple of (is_available, module_or_attribute)
        - is_available: True if import succeeded
        - module_or_attribute: The imported module/attribute or None

    Example:
        >>> OPENAI_AVAILABLE, OpenAI = optional_import('openai', attribute='OpenAI')
        >>> if OPENAI_AVAILABLE:
        ...     client = OpenAI()
    """
    if package_name is None:
        package_name = module_name

    try:
        module = import_module(module_name)
        if attribute:
            return True, getattr(module, attribute)
        return True, module
    except ImportError:
        return False, None


def require_dependency(
    module_name: str,
    package_name: str | None = None,
    feature: str = "",
) -> Any:
    """
    Import a module, raising MissingDependencyError if not available.

    Use this when a feature absolutely requires a dependency and should
    fail with a clear error message if it's missing.

    Args:
        module_name: The module to import
        package_name: The pip package name for install hints
        feature: Description of what feature requires this dependency

    Returns:
        The imported module

    Raises:
        MissingDependencyError: If the module cannot be imported

    Example:
        >>> def process_audio():
        ...     np = require_dependency('numpy', feature='audio processing')
        ...     return np.array([1, 2, 3])
    """
    if package_name is None:
        package_name = module_name

    try:
        return import_module(module_name)
    except ImportError:
        raise MissingDependencyError(package_name, feature) from None


class DependencyChecker:
    """
    Centralized checker for optional dependencies.

    Use this class to check dependency availability without importing
    multiple times throughout the codebase.

    Example:
        >>> deps = DependencyChecker()
        >>> if deps.has_openai:
        ...     from openai import OpenAI
        ...     client = OpenAI()
    """

    def __init__(self) -> None:
        self._cache: dict[str, bool] = {}

    def check(self, module_name: str) -> bool:
        """Check if a module is available, caching the result."""
        if module_name not in self._cache:
            self._cache[module_name] = self._do_check(module_name)
        return self._cache[module_name]

    def _do_check(self, module_name: str) -> bool:
        try:
            import_module(module_name)
            return True
        except ImportError:
            return False

    # Common dependency properties
    @property
    def has_numpy(self) -> bool:
        return self.check("numpy")

    @property
    def has_scipy(self) -> bool:
        return self.check("scipy")

    @property
    def has_openai(self) -> bool:
        return self.check("openai")

    @property
    def has_pydantic(self) -> bool:
        return self.check("pydantic")

    @property
    def has_websockets(self) -> bool:
        return self.check("websockets")

    @property
    def has_aiortc(self) -> bool:
        return self.check("aiortc")

    @property
    def has_pyaudio(self) -> bool:
        return self.check("pyaudio")

    @property
    def has_sounddevice(self) -> bool:
        return self.check("sounddevice")

    @property
    def has_psutil(self) -> bool:
        return self.check("psutil")

    @property
    def has_webrtcvad(self) -> bool:
        return self.check("webrtcvad")

    @property
    def has_torch(self) -> bool:
        return self.check("torch")


# Global instance for convenience
deps = DependencyChecker()

# Pre-checked common dependencies (useful for module-level imports)
NUMPY_AVAILABLE, np = optional_import("numpy")
SCIPY_AVAILABLE, scipy = optional_import("scipy")
OPENAI_AVAILABLE, openai = optional_import("openai")
PYDANTIC_AVAILABLE, pydantic = optional_import("pydantic")
WEBSOCKETS_AVAILABLE, websockets = optional_import("websockets")
AIORTC_AVAILABLE, aiortc = optional_import("aiortc")
PSUTIL_AVAILABLE, psutil = optional_import("psutil")


# --- Pydantic Fallback Implementation ---
# Provides minimal BaseModel when pydantic is not installed


class _FallbackBaseModel:
    """Minimal BaseModel fallback when pydantic is not available."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "_FallbackBaseModel":
        return cls(**data)


def _fallback_field(default: Any = None, **kwargs: Any) -> Any:
    """Minimal Field fallback when pydantic is not available."""
    return default


def _fallback_field_validator(*args: Any, **kwargs: Any) -> Any:
    """Minimal field_validator fallback when pydantic is not available."""

    def decorator(func: Any) -> Any:
        return func

    return decorator


# Export pydantic components with fallbacks
if PYDANTIC_AVAILABLE:
    BaseModel = pydantic.BaseModel  # type: ignore[misc]
    Field = pydantic.Field  # type: ignore[misc]
    field_validator = pydantic.field_validator  # type: ignore[misc]
else:
    BaseModel = _FallbackBaseModel
    Field = _fallback_field
    field_validator = _fallback_field_validator
