"""Health check module for production readiness.

This module provides health check functionality for monitoring the
status of various application components including configuration,
API keys, and required directories.
"""

import os
from pathlib import Path
from typing import Any


def check_health() -> dict[str, Any]:
    """Check application health status.

    Performs health checks on various application components:
    - Configuration loading
    - API key availability
    - Directory writability

    Returns:
        Dict with health status of each component:
        - config: Can load configuration (bool)
        - api_key: API key is set (bool)
        - directories: Required directories are writable (bool)
        - details: Additional information about each check
        - healthy: Overall health status (bool)
    """
    results: dict[str, Any] = {
        "config": False,
        "api_key": False,
        "directories": False,
        "details": {},
        "healthy": False,
    }

    # Check config loading
    config_healthy, config_details = _check_config()
    results["config"] = config_healthy
    results["details"]["config"] = config_details

    # Check API key
    api_key_healthy, api_key_details = _check_api_key()
    results["api_key"] = api_key_healthy
    results["details"]["api_key"] = api_key_details

    # Check directories
    directories_healthy, directories_details = _check_directories()
    results["directories"] = directories_healthy
    results["details"]["directories"] = directories_details

    # Overall health
    results["healthy"] = all([
        results["config"],
        results["api_key"],
        results["directories"],
    ])

    return results


def _check_config() -> tuple[bool, dict[str, Any]]:
    """Check if configuration can be loaded.

    Returns:
        Tuple of (healthy: bool, details: dict)
    """
    details: dict[str, Any] = {"checks": []}

    try:
        from src.config import settings

        # Try to access a few key settings
        _ = settings.OPENAI_API_KEY
        _ = settings.LOG_LEVEL

        details["checks"].append("Settings loaded successfully")
        return True, details
    except ImportError as e:
        details["checks"].append(f"Failed to import settings: {e}")
        return False, details
    except Exception as e:
        details["checks"].append(f"Configuration error: {e}")
        return False, details


def _check_api_key() -> tuple[bool, dict[str, Any]]:
    """Check if OpenAI API key is set.

    Returns:
        Tuple of (healthy: bool, details: dict)
    """
    details: dict[str, Any] = {"checks": []}

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if api_key:
        # Basic validation - check it looks like an API key
        if api_key.startswith("sk-") or len(api_key) > 20:
            details["checks"].append("API key is set")
            details["key_prefix"] = api_key[:7] + "..." if len(api_key) > 10 else "***"
            return True, details
        else:
            details["checks"].append("API key format appears invalid")
            return False, details
    else:
        details["checks"].append("API key not found in environment")
        return False, details


def _check_directories() -> tuple[bool, dict[str, Any]]:
    """Check if required directories exist and are writable.

    Returns:
        Tuple of (healthy: bool, details: dict)
    """
    details: dict[str, Any] = {"checks": [], "directories": {}}

    # Directories to check
    required_dirs: list[str] = [
        "logs",
        "models",
        "audio_output",
    ]

    all_healthy = True

    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dir_info: dict[str, Any] = {}

        # Check if directory exists
        if not dir_path.exists():
            dir_info["exists"] = False
            dir_info["writable"] = False
            details["checks"].append(f"Directory '{dir_name}' does not exist")
            all_healthy = False
        else:
            dir_info["exists"] = True

            # Check if writable
            if os.access(dir_path, os.W_OK):
                dir_info["writable"] = True
                details["checks"].append(f"Directory '{dir_name}' is writable")
            else:
                dir_info["writable"] = False
                details["checks"].append(f"Directory '{dir_name}' is not writable")
                all_healthy = False

        details["directories"][dir_name] = dir_info

    return all_healthy, details


def get_component_status(component: str) -> dict[str, Any] | None:
    """Get health status for a specific component.

    Args:
        component: Component name ('config', 'api_key', 'directories')

    Returns:
        Dict with component health status or None if component not found.
    """
    health = check_health()

    if component == "config":
        return {
            "healthy": health["config"],
            "details": health["details"]["config"],
        }
    elif component == "api_key":
        return {
            "healthy": health["api_key"],
            "details": health["details"]["api_key"],
        }
    elif component == "directories":
        return {
            "healthy": health["directories"],
            "details": health["details"]["directories"],
        }
    else:
        return None


__all__ = [
    "check_health",
    "get_component_status",
]
