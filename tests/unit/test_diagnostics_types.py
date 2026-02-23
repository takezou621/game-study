"""Tests for diagnostics types module."""

import time

import pytest

from diagnostics.diagnostics_types import (
    CheckResult,
    CheckStatus,
    DeviceInfo,
    SystemComponent,
    SystemInfo,
)


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_has_all_statuses(self) -> None:
        """Test that all expected statuses exist."""
        assert CheckStatus.PASSED.value == "passed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.WARNING.value == "warning"
        assert CheckStatus.SKIPPED.value == "skipped"

    def test_all_statuses_are_unique(self) -> None:
        """Test that all statuses have unique values."""
        values = [status.value for status in CheckStatus]
        assert len(values) == len(set(values))


class TestSystemComponent:
    """Tests for SystemComponent enum."""

    def test_has_all_components(self) -> None:
        """Test that all expected components exist."""
        expected = [
            "microphone",
            "speaker",
            "network",
            "memory",
            "cpu",
            "disk",
            "permissions",
        ]
        actual = [comp.value for comp in SystemComponent]
        assert set(actual) == set(expected)

    def test_all_components_are_unique(self) -> None:
        """Test that all components have unique values."""
        values = [comp.value for comp in SystemComponent]
        assert len(values) == len(set(values))


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic CheckResult."""
        result = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.PASSED,
            message="Microphone is working",
        )
        assert result.component == SystemComponent.MICROPHONE
        assert result.status == CheckStatus.PASSED
        assert result.message == "Microphone is working"
        assert result.remediation is None

    def test_auto_timestamp(self) -> None:
        """Test that timestamp is auto-generated."""
        before = time.time()
        result = CheckResult(
            component=SystemComponent.CPU,
            status=CheckStatus.PASSED,
            message="CPU OK",
        )
        after = time.time()

        assert result.timestamp is not None
        assert before <= result.timestamp <= after

    def test_with_details(self) -> None:
        """Test CheckResult with details."""
        result = CheckResult(
            component=SystemComponent.MEMORY,
            status=CheckStatus.WARNING,
            message="Low memory",
            details={"free_gb": 0.5, "used_percent": 95.0},
        )
        assert result.details["free_gb"] == 0.5
        assert result.details["used_percent"] == 95.0

    def test_with_remediation(self) -> None:
        """Test CheckResult with remediation."""
        result = CheckResult(
            component=SystemComponent.NETWORK,
            status=CheckStatus.FAILED,
            message="No network",
            remediation="Check your internet connection",
        )
        assert result.remediation == "Check your internet connection"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        result = CheckResult(
            component=SystemComponent.DISK,
            status=CheckStatus.WARNING,
            message="Low disk space",
            details={"free_gb": 10.0},
            remediation="Delete unused files",
        )
        d = result.to_dict()

        assert d["component"] == "disk"
        assert d["status"] == "warning"
        assert d["message"] == "Low disk space"
        assert d["details"]["free_gb"] == 10.0
        assert d["remediation"] == "Delete unused files"
        assert "timestamp" in d


class TestSystemInfo:
    """Tests for SystemInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic SystemInfo."""
        info = SystemInfo(
            os="Darwin",
            os_version="23.0.0",
            python_version="3.11.0",
        )
        assert info.os == "Darwin"
        assert info.os_version == "23.0.0"
        assert info.python_version == "3.11.0"
        assert info.cpu_count is None
        assert info.total_memory_gb is None

    def test_with_optional_fields(self) -> None:
        """Test SystemInfo with optional fields."""
        info = SystemInfo(
            os="Linux",
            os_version="6.0",
            python_version="3.10.0",
            cpu_count=8,
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            disk_usage_percent=45.5,
        )
        assert info.cpu_count == 8
        assert info.total_memory_gb == 16.0
        assert info.available_memory_gb == 8.0
        assert info.disk_usage_percent == 45.5

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        info = SystemInfo(
            os="Windows",
            os_version="10",
            python_version="3.9.0",
            cpu_count=4,
        )
        d = info.to_dict()

        assert d["os"] == "Windows"
        assert d["os_version"] == "10"
        assert d["python_version"] == "3.9.0"
        assert d["cpu_count"] == 4


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_input_device(self) -> None:
        """Test creating an input device info."""
        device = DeviceInfo(
            name="Microphone",
            index=0,
            channels=1,
            sample_rate=48000,
            is_input=True,
            is_output=False,
        )
        assert device.name == "Microphone"
        assert device.index == 0
        assert device.is_input is True
        assert device.is_output is False
        assert device.is_default is False

    def test_output_device(self) -> None:
        """Test creating an output device info."""
        device = DeviceInfo(
            name="Speakers",
            index=1,
            channels=2,
            sample_rate=44100,
            is_input=False,
            is_output=True,
            is_default=True,
        )
        assert device.name == "Speakers"
        assert device.is_input is False
        assert device.is_output is True
        assert device.is_default is True

    def test_default_values(self) -> None:
        """Test default values."""
        device = DeviceInfo(
            name="Device",
            index=0,
            channels=2,
            sample_rate=16000,
            is_input=True,
            is_output=False,
        )
        assert device.is_default is False
