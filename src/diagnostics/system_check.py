"""System diagnostics for device connectivity and performance.

This module provides system-level diagnostics including:
- Microphone and speaker device checks
- Network connectivity tests
- Performance monitoring
- Resource availability checks
"""

import asyncio
import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from utils.exceptions import GameStudyError
from utils.logger import get_logger

logger = get_logger(__name__)


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
            "remediation": self.remediation
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
            "disk_usage_percent": self.disk_usage_percent
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


class SystemDiagnosticsError(GameStudyError):
    """Exception raised for system diagnostics errors."""

    def __init__(
        self,
        message: str,
        component: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        context = context or {}
        context.update({"component": component})
        super().__init__(message, context=context, cause=cause)
        self.component = component


class AudioDeviceChecker:
    """Checks audio device availability and configuration."""

    def __init__(self):
        """Initialize audio device checker."""
        self.pyaudio = None
        self.sounddevice = None

        # Try to import audio libraries
        try:
            import pyaudio
            self.pyaudio = pyaudio
        except ImportError:
            pass

        try:
            import sounddevice as sd
            self.sounddevice = sd
        except ImportError:
            pass

    def get_available_devices(self) -> tuple[list[DeviceInfo], list[DeviceInfo]]:
        """Get available input and output audio devices.

        Returns:
            Tuple of (input_devices, output_devices)
        """
        input_devices = []
        output_devices = []

        if self.pyaudio:
            try:
                p = self.pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)

                    if info['maxInputChannels'] > 0:
                        input_devices.append(DeviceInfo(
                            name=info['name'],
                            index=i,
                            channels=int(info['maxInputChannels']),
                            sample_rate=int(info['defaultSampleRate']),
                            is_input=True,
                            is_output=False
                        ))

                    if info['maxOutputChannels'] > 0:
                        output_devices.append(DeviceInfo(
                            name=info['name'],
                            index=i,
                            channels=int(info['maxOutputChannels']),
                            sample_rate=int(info['defaultSampleRate']),
                            is_input=False,
                            is_output=True
                        ))

                p.terminate()
            except Exception as e:
                logger.warning(f"Error getting PyAudio devices: {e}")

        elif self.sounddevice:
            try:
                devices = self.sounddevice.query_devices()

                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        input_devices.append(DeviceInfo(
                            name=device['name'],
                            index=i,
                            channels=int(device['max_input_channels']),
                            sample_rate=int(device['default_samplerate']),
                            is_input=True,
                            is_output=False
                        ))

                    if device['max_output_channels'] > 0:
                        output_devices.append(DeviceInfo(
                            name=device['name'],
                            index=i,
                            channels=int(device['max_output_channels']),
                            sample_rate=int(device['default_samplerate']),
                            is_input=False,
                            is_output=True
                        ))
            except Exception as e:
                logger.warning(f"Error getting sounddevice devices: {e}")

        return input_devices, output_devices

    async def check_microphone(self, device_index: int | None = None) -> CheckResult:
        """Check if microphone is accessible.

        Args:
            device_index: Specific device index to check, or None for default

        Returns:
            CheckResult with status and details
        """
        input_devices, _ = self.get_available_devices()

        if not input_devices:
            return CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.FAILED,
                message="No microphone devices found",
                remediation="Connect a microphone or check audio settings"
            )

        # Find device to check
        device = None
        if device_index is not None:
            for d in input_devices:
                if d.index == device_index:
                    device = d
                    break
        else:
            # Use first available device
            device = input_devices[0]

        if device is None:
            return CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.FAILED,
                message=f"Microphone device {device_index} not found",
                remediation="Check device index in configuration"
            )

        # Try to open the device
        try:
            if self.pyaudio:
                return await self._check_pyaudio_microphone(device)
            elif self.sounddevice:
                return await self._check_sounddevice_microphone(device)
            else:
                return CheckResult(
                    component=SystemComponent.MICROPHONE,
                    status=CheckStatus.WARNING,
                    message="No audio library available for testing",
                    remediation="Install pyaudio or sounddevice: pip install pyaudio"
                )

        except Exception as e:
            return CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.FAILED,
                message=f"Microphone check failed: {e}",
                remediation="Check microphone permissions and connections"
            )

    async def _check_pyaudio_microphone(self, device: DeviceInfo) -> CheckResult:
        """Check microphone using PyAudio."""
        try:
            loop = asyncio.get_event_loop()
            p = self.pyaudio.PyAudio()

            def test_stream():
                try:
                    stream = p.open(
                        format=self.pyaudio.paInt16,
                        channels=1,
                        rate=device.sample_rate,
                        input=True,
                        input_device_index=device.index,
                        frames_per_buffer=1024
                    )
                    # Read a small chunk to verify
                    data = stream.read(1024, exception_on_overflow=False)
                    stream.close()
                    return True
                except Exception:
                    raise

            await loop.run_in_executor(None, test_stream)
            p.terminate()

            return CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.PASSED,
                message=f"Microphone '{device.name}' is working",
                details={
                    "device_name": device.name,
                    "sample_rate": device.sample_rate,
                    "channels": device.channels
                }
            )

        except OSError as e:
            return CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.FAILED,
                message=f"Cannot access microphone: {e}",
                remediation="Check microphone permissions (macOS: System Preferences > Privacy)"
            )
        except Exception:
            raise

    async def _check_sounddevice_microphone(self, device: DeviceInfo) -> CheckResult:
        """Check microphone using sounddevice."""
        try:
            loop = asyncio.get_event_loop()

            def test_record():
                # Record a short sample
                import sounddevice as sd
                data = sd.rec(
                    1024,
                    samplerate=device.sample_rate,
                    channels=1,
                    dtype='int16',
                    device=device.index
                )
                sd.wait()
                return True

            await loop.run_in_executor(None, test_record)

            return CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.PASSED,
                message=f"Microphone '{device.name}' is working",
                details={
                    "device_name": device.name,
                    "sample_rate": device.sample_rate,
                    "channels": device.channels
                }
            )

        except Exception:
            raise

    async def check_speaker(self, device_index: int | None = None) -> CheckResult:
        """Check if speaker is accessible.

        Args:
            device_index: Specific device index to check, or None for default

        Returns:
            CheckResult with status and details
        """
        _, output_devices = self.get_available_devices()

        if not output_devices:
            return CheckResult(
                component=SystemComponent.SPEAKER,
                status=CheckStatus.FAILED,
                message="No speaker devices found",
                remediation="Connect speakers or check audio settings"
            )

        # Find device to check
        device = None
        if device_index is not None:
            for d in output_devices:
                if d.index == device_index:
                    device = d
                    break
        else:
            device = output_devices[0]

        if device is None:
            return CheckResult(
                component=SystemComponent.SPEAKER,
                status=CheckStatus.FAILED,
                message=f"Speaker device {device_index} not found",
                details={"device_index": device_index}
            )

        # Try to open the device
        try:
            if self.pyaudio:
                return await self._check_pyaudio_speaker(device)
            elif self.sounddevice:
                return CheckResult(
                    component=SystemComponent.SPEAKER,
                    status=CheckStatus.WARNING,
                    message="Speaker available but audio test not performed",
                    details={
                        "device_name": device.name,
                        "sample_rate": device.sample_rate
                    }
                )
            else:
                return CheckResult(
                    component=SystemComponent.SPEAKER,
                    status=CheckStatus.WARNING,
                    message="No audio library available for testing",
                    remediation="Install pyaudio or sounddevice"
                )

        except Exception as e:
            return CheckResult(
                component=SystemComponent.SPEAKER,
                status=CheckStatus.FAILED,
                message=f"Speaker check failed: {e}",
                remediation="Check speaker connections and volume"
            )

    async def _check_pyaudio_speaker(self, device: DeviceInfo) -> CheckResult:
        """Check speaker using PyAudio."""
        try:
            loop = asyncio.get_event_loop()
            p = self.pyaudio.PyAudio()

            def test_stream():
                try:
                    stream = p.open(
                        format=self.pyaudio.paInt16,
                        channels=1,
                        rate=device.sample_rate,
                        output=True,
                        output_device_index=device.index,
                        frames_per_buffer=1024
                    )
                    # Write a silent chunk to verify
                    silence = b'\x00\x00' * 1024
                    stream.write(silence)
                    stream.close()
                    return True
                except Exception:
                    raise

            await loop.run_in_executor(None, test_stream)
            p.terminate()

            return CheckResult(
                component=SystemComponent.SPEAKER,
                status=CheckStatus.PASSED,
                message=f"Speaker '{device.name}' is working",
                details={
                    "device_name": device.name,
                    "sample_rate": device.sample_rate,
                    "channels": device.channels
                }
            )

        except Exception:
            raise


class NetworkChecker:
    """Checks network connectivity and performance."""

    def __init__(self, test_hosts: list[str] | None = None):
        """Initialize network checker.

        Args:
            test_hosts: List of hosts to test connectivity
        """
        self.test_hosts = test_hosts or [
            "api.openai.com",
            "8.8.8.8",  # Google DNS
            "1.1.1.1"   # Cloudflare DNS
        ]

    async def check_connectivity(self, host: str, timeout: float = 5.0) -> tuple[bool, float]:
        """Check connectivity to a host.

        Args:
            host: Host to check
            timeout: Timeout in seconds

        Returns:
            Tuple of (connected, latency_ms)
        """
        try:
            loop = asyncio.get_event_loop()
            start_time = time.time()

            # Try to create a connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, 443),
                timeout=timeout
            )

            latency_ms = (time.time() - start_time) * 1000

            writer.close()
            await writer.wait_closed()

            return True, latency_ms

        except (asyncio.TimeoutError, OSError):
            return False, 0.0

    async def check_network(self) -> CheckResult:
        """Perform comprehensive network check.

        Returns:
            CheckResult with network status
        """
        results = []
        total_latency = 0.0
        connected_count = 0

        for host in self.test_hosts:
            connected, latency = await self.check_connectivity(host)
            results.append({
                "host": host,
                "connected": connected,
                "latency_ms": latency
            })

            if connected:
                connected_count += 1
                total_latency += latency

        # Determine overall status
        if connected_count == 0:
            status = CheckStatus.FAILED
            message = "No network connectivity detected"
            remediation = "Check internet connection and firewall settings"
        elif connected_count < len(self.test_hosts):
            status = CheckStatus.WARNING
            message = f"Partial network connectivity ({connected_count}/{len(self.test_hosts)} hosts)"
            remediation = "Some services may be unavailable"
        else:
            avg_latency = total_latency / connected_count
            if avg_latency > 500:
                status = CheckStatus.WARNING
                message = f"Network connected but high latency ({avg_latency:.1f}ms)"
                remediation = "Consider using wired connection for better performance"
            else:
                status = CheckStatus.PASSED
                message = f"Network connected ({avg_latency:.1f}ms average latency)"
                remediation = None

        return CheckResult(
            component=SystemComponent.NETWORK,
            status=status,
            message=message,
            details={
                "hosts_tested": results,
                "connected_count": connected_count,
                "average_latency_ms": total_latency / max(connected_count, 1)
            },
            remediation=remediation
        )

    async def check_api_reachable(self, api_url: str = "api.openai.com") -> CheckResult:
        """Check if specific API is reachable.

        Args:
            api_url: API host to check

        Returns:
            CheckResult with API reachability status
        """
        connected, latency = await self.check_connectivity(api_url, timeout=10.0)

        if connected:
            return CheckResult(
                component=SystemComponent.NETWORK,
                status=CheckStatus.PASSED,
                message=f"API reachable ({latency:.1f}ms latency)",
                details={"api_host": api_url, "latency_ms": latency}
            )
        else:
            return CheckResult(
                component=SystemComponent.NETWORK,
                status=CheckStatus.FAILED,
                message=f"API unreachable: {api_url}",
                details={"api_host": api_url},
                remediation="Check internet connection and API status"
            )


class PerformanceChecker:
    """Checks system performance and resources."""

    def __init__(self, min_memory_gb: float = 1.0, min_disk_percent: float = 90.0):
        """Initialize performance checker.

        Args:
            min_memory_gb: Minimum required free memory in GB
            min_disk_percent: Maximum allowed disk usage percentage
        """
        self.min_memory_gb = min_memory_gb
        self.min_disk_percent = min_disk_percent

    def get_system_info(self) -> SystemInfo:
        """Get system information.

        Returns:
            SystemInfo object
        """
        info = SystemInfo(
            os=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version()
        )

        if PSUTIL_AVAILABLE:
            try:
                info.cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory()
                info.total_memory_gb = memory.total / (1024**3)
                info.available_memory_gb = memory.available / (1024**3)

                disk = psutil.disk_usage('/')
                info.disk_usage_percent = disk.percent
            except Exception as e:
                logger.warning(f"Error getting system info: {e}")

        return info

    async def check_memory(self) -> CheckResult:
        """Check available memory.

        Returns:
            CheckResult with memory status
        """
        if not PSUTIL_AVAILABLE:
            return CheckResult(
                component=SystemComponent.MEMORY,
                status=CheckStatus.SKIPPED,
                message="Memory check skipped (psutil not available)"
            )

        try:
            loop = asyncio.get_event_loop()
            memory = await loop.run_in_executor(None, psutil.virtual_memory)
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            used_percent = memory.percent

            if available_gb < self.min_memory_gb:
                return CheckResult(
                    component=SystemComponent.MEMORY,
                    status=CheckStatus.WARNING if used_percent < 95 else CheckStatus.FAILED,
                    message=f"Low memory: {available_gb:.2f}GB free ({used_percent:.1f}% used)",
                    details={
                        "available_gb": available_gb,
                        "total_gb": total_gb,
                        "used_percent": used_percent
                    },
                    remediation="Close unnecessary applications to free memory"
                )
            else:
                return CheckResult(
                    component=SystemComponent.MEMORY,
                    status=CheckStatus.PASSED,
                    message=f"Memory OK: {available_gb:.2f}GB free ({used_percent:.1f}% used)",
                    details={
                        "available_gb": available_gb,
                        "total_gb": total_gb,
                        "used_percent": used_percent
                    }
                )

        except Exception as e:
            return CheckResult(
                component=SystemComponent.MEMORY,
                status=CheckStatus.FAILED,
                message=f"Memory check failed: {e}"
            )

    async def check_cpu(self) -> CheckResult:
        """Check CPU usage.

        Returns:
            CheckResult with CPU status
        """
        if not PSUTIL_AVAILABLE:
            return CheckResult(
                component=SystemComponent.CPU,
                status=CheckStatus.SKIPPED,
                message="CPU check skipped (psutil not available)"
            )

        try:
            loop = asyncio.get_event_loop()
            cpu_percent = await loop.run_in_executor(
                None,
                lambda: psutil.cpu_percent(interval=0.5)
            )

            cpu_count = psutil.cpu_count()

            if cpu_percent > 90:
                status = CheckStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
                remediation = "Close unnecessary applications"
            elif cpu_percent > 95:
                status = CheckStatus.FAILED
                message = f"Very high CPU usage: {cpu_percent:.1f}%"
                remediation = "Close applications or upgrade hardware"
            else:
                status = CheckStatus.PASSED
                message = f"CPU usage OK: {cpu_percent:.1f}%"
                remediation = None

            return CheckResult(
                component=SystemComponent.CPU,
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count
                },
                remediation=remediation
            )

        except Exception as e:
            return CheckResult(
                component=SystemComponent.CPU,
                status=CheckStatus.FAILED,
                message=f"CPU check failed: {e}"
            )

    async def check_disk(self) -> CheckResult:
        """Check disk space.

        Returns:
            CheckResult with disk status
        """
        if not PSUTIL_AVAILABLE:
            return CheckResult(
                component=SystemComponent.DISK,
                status=CheckStatus.SKIPPED,
                message="Disk check skipped (psutil not available)"
            )

        try:
            loop = asyncio.get_event_loop()
            disk = await loop.run_in_executor(
                None,
                lambda: psutil.disk_usage('/')
            )

            used_percent = disk.percent
            free_gb = disk.free / (1024**3)

            if used_percent > self.min_disk_percent:
                return CheckResult(
                    component=SystemComponent.DISK,
                    status=CheckStatus.FAILED if used_percent > 98 else CheckStatus.WARNING,
                    message=f"Low disk space: {free_gb:.2f}GB free ({used_percent:.1f}% used)",
                    details={
                        "free_gb": free_gb,
                        "used_percent": used_percent
                    },
                    remediation="Free up disk space by removing unnecessary files"
                )
            else:
                return CheckResult(
                    component=SystemComponent.DISK,
                    status=CheckStatus.PASSED,
                    message=f"Disk space OK: {free_gb:.2f}GB free ({used_percent:.1f}% used)",
                    details={
                        "free_gb": free_gb,
                        "used_percent": used_percent
                    }
                )

        except Exception as e:
            return CheckResult(
                component=SystemComponent.DISK,
                status=CheckStatus.FAILED,
                message=f"Disk check failed: {e}"
            )


class PermissionChecker:
    """Checks application permissions."""

    @staticmethod
    def check_microphone_permission() -> CheckResult:
        """Check microphone access permission.

        Returns:
            CheckResult with permission status
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            # On macOS, we can't directly check permission without actually trying to access
            # This is a best-effort check
            return CheckResult(
                component=SystemComponent.PERMISSIONS,
                status=CheckStatus.WARNING,
                message="Microphone permission status unknown (requires runtime check)",
                remediation="Grant microphone access in System Preferences > Privacy & Security > Microphone"
            )
        elif system == "Linux":
            # Check if user is in audio group
            try:
                import grp
                audio_group = grp.getgrnam("audio")
                if os.getuid() in audio_group.gr_mem:
                    return CheckResult(
                        component=SystemComponent.PERMISSIONS,
                        status=CheckStatus.PASSED,
                        message="User is in audio group"
                    )
                else:
                    return CheckResult(
                        component=SystemComponent.PERMISSIONS,
                        status=CheckStatus.WARNING,
                        message="User not in audio group",
                        remediation="Add user to audio group: sudo usermod -a -G audio $USER"
                    )
            except Exception:
                pass

        return CheckResult(
            component=SystemComponent.PERMISSIONS,
            status=CheckStatus.SKIPPED,
            message="Permission check not available on this platform"
        )

    @staticmethod
    def check_screen_recording_permission() -> CheckResult:
        """Check screen recording permission (for screen capture).

        Returns:
            CheckResult with permission status
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            return CheckResult(
                component=SystemComponent.PERMISSIONS,
                status=CheckStatus.WARNING,
                message="Screen recording permission requires runtime check",
                remediation="Grant screen recording in System Preferences > Privacy & Security > Screen Recording"
            )

        return CheckResult(
            component=SystemComponent.PERMISSIONS,
            status=CheckStatus.SKIPPED,
            message="Screen recording permission check not needed on this platform"
        )


class SystemDiagnostics:
    """Main system diagnostics class combining all checks."""

    def __init__(
        self,
        min_memory_gb: float = 1.0,
        min_disk_percent: float = 90.0,
        test_hosts: list[str] | None = None
    ):
        """Initialize system diagnostics.

        Args:
            min_memory_gb: Minimum required free memory in GB
            min_disk_percent: Maximum allowed disk usage percentage
            test_hosts: Hosts to test for network connectivity
        """
        self.audio_checker = AudioDeviceChecker()
        self.network_checker = NetworkChecker(test_hosts)
        self.performance_checker = PerformanceChecker(min_memory_gb, min_disk_percent)
        self.permission_checker = PermissionChecker()

        self._check_results: list[CheckResult] = []

    async def run_all_checks(self) -> list[CheckResult]:
        """Run all diagnostic checks.

        Returns:
            List of CheckResult objects
        """
        results = []

        # Audio checks
        mic_result = await self.audio_checker.check_microphone()
        results.append(mic_result)

        speaker_result = await self.audio_checker.check_speaker()
        results.append(speaker_result)

        # Network checks
        network_result = await self.network_checker.check_network()
        results.append(network_result)

        # Performance checks
        memory_result = await self.performance_checker.check_memory()
        results.append(memory_result)

        cpu_result = await self.performance_checker.check_cpu()
        results.append(cpu_result)

        disk_result = await self.performance_checker.check_disk()
        results.append(disk_result)

        # Permission checks
        mic_perm_result = self.permission_checker.check_microphone_permission()
        results.append(mic_perm_result)

        screen_perm_result = self.permission_checker.check_screen_recording_permission()
        results.append(screen_perm_result)

        self._check_results = results
        return results

    async def check_audio_devices(self) -> list[CheckResult]:
        """Check only audio devices.

        Returns:
            List of CheckResult objects for audio devices
        """
        mic_result = await self.audio_checker.check_microphone()
        speaker_result = await self.audio_checker.check_speaker()
        return [mic_result, speaker_result]

    async def check_network_only(self) -> CheckResult:
        """Check only network connectivity.

        Returns:
            Network CheckResult
        """
        return await self.network_checker.check_network()

    async def check_api_reachable(self, api_url: str = "api.openai.com") -> CheckResult:
        """Check if specific API is reachable.

        Args:
            api_url: API host to check

        Returns:
            CheckResult with API reachability status
        """
        return await self.network_checker.check_api_reachable(api_url)

    def get_system_info(self) -> SystemInfo:
        """Get system information.

        Returns:
            SystemInfo object
        """
        return self.performance_checker.get_system_info()

    def get_check_results(self) -> list[CheckResult]:
        """Get results from last check run.

        Returns:
            List of CheckResult objects
        """
        return self._check_results

    def get_failed_checks(self) -> list[CheckResult]:
        """Get only failed checks from last run.

        Returns:
            List of failed CheckResult objects
        """
        return [r for r in self._check_results if r.status == CheckStatus.FAILED]

    def get_warning_checks(self) -> list[CheckResult]:
        """Get only warning checks from last run.

        Returns:
            List of warning CheckResult objects
        """
        return [r for r in self._check_results if r.status == CheckStatus.WARNING]


async def create_system_diagnostics(
    min_memory_gb: float = 1.0,
    min_disk_percent: float = 90.0,
    test_hosts: list[str] | None = None
) -> SystemDiagnostics:
    """Create a SystemDiagnostics instance.

    Args:
        min_memory_gb: Minimum required free memory in GB
        min_disk_percent: Maximum allowed disk usage percentage
        test_hosts: Hosts to test for network connectivity

    Returns:
        Configured SystemDiagnostics instance
    """
    return SystemDiagnostics(
        min_memory_gb=min_memory_gb,
        min_disk_percent=min_disk_percent,
        test_hosts=test_hosts
    )
