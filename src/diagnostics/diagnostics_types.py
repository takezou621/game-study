"""Common data types for diagnostics modules.

This module contains dataclasses and enums shared across diagnostic components,
separated for better code organization and reusability.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(Enum):
    """Status of a diagnostic check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class SystemComponent(Enum):
    """System components that can be checked."""

    MICROPHONE = "microphone"
    SPEAKER = "speaker"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    PERMISSIONS = "permissions"


@dataclass
class CheckResult:
    """Result of a diagnostic check."""

    component: SystemComponent
    status: CheckStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
            "remediation": self.remediation,
        }


@dataclass
class SystemInfo:
    """System information."""

    os: str
    os_version: str
    python_version: str
    cpu_count: int | None = None
    total_memory_gb: float | None = None
    available_memory_gb: float | None = None
    disk_usage_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "os": self.os,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "disk_usage_percent": self.disk_usage_percent,
        }


@dataclass
class DeviceInfo:
    """Audio device information."""

    name: str
    index: int
    channels: int
    sample_rate: int
    is_input: bool
    is_output: bool
    is_default: bool = False
