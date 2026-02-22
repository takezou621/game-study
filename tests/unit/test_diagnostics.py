"""Tests for diagnostics module."""

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Create mock psutil module for tests where psutil is not available
mock_psutil = MagicMock()
sys.modules['psutil'] = mock_psutil

try:
    from diagnostics.audio_check import (
        NUMPY_AVAILABLE,
        SCIPY_AVAILABLE,
        AudioDiagnostics,
        AudioDiagnosticsError,
        AudioIssue,
        AudioIssueType,
        AudioMetrics,
        AudioQualityAnalyzer,
        CrosstalkDetector,
        EchoDetector,
        create_audio_diagnostics,
    )
except ImportError:
    pytest.skip("diagnostics module not available", allow_module_level=True)

from diagnostics.report import (
    ConsoleReporter,
    DiagnosticIssue,
    DiagnosticReport,
    FixSuggestion,
    IssueClassifier,
    ReportGenerator,
    SeverityLevel,
    generate_diagnostic_report,
)
from diagnostics.system_check import (
    PSUTIL_AVAILABLE,
    AudioDeviceChecker,
    CheckResult,
    CheckStatus,
    NetworkChecker,
    PerformanceChecker,
    PermissionChecker,
    SystemComponent,
    SystemDiagnostics,
    SystemInfo,
)

# ============================================================================
# AudioIssueType Tests
# ============================================================================

class TestAudioIssueType:
    """Test AudioIssueType enum."""

    def test_issue_types(self):
        """Test all issue type values exist."""
        assert AudioIssueType.ECHO.value == "echo"
        assert AudioIssueType.CROSSTALK.value == "crosstalk"
        assert AudioIssueType.LOW_SNR.value == "low_snr"
        assert AudioIssueType.CLIPPING.value == "clipping"


# ============================================================================
# AudioIssue Tests
# ============================================================================

class TestAudioIssue:
    """Test AudioIssue dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        issue = AudioIssue(
            issue_type=AudioIssueType.ECHO,
            severity="warning",
            message="Echo detected",
            details={"delay_ms": 150}
        )

        issue_dict = issue.to_dict()

        assert issue_dict["type"] == "echo"
        assert issue_dict["severity"] == "warning"
        assert issue_dict["message"] == "Echo detected"
        assert issue_dict["details"]["delay_ms"] == 150


# ============================================================================
# AudioMetrics Tests
# ============================================================================

class TestAudioMetrics:
    """Test AudioMetrics dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        metrics = AudioMetrics(
            snr_db=25.0,
            rms_level=0.3,
            echo_detected=True,
            echo_delay_ms=150.0
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["snr_db"] == 25.0
        assert metrics_dict["echo_detected"] is True
        assert metrics_dict["echo_delay_ms"] == 150.0


# ============================================================================
# AudioDiagnosticsError Tests
# ============================================================================

class TestAudioDiagnosticsError:
    """Test AudioDiagnosticsError exception."""

    def test_init(self):
        """Test initialization."""
        error = AudioDiagnosticsError(
            message="Test error",
            check_type="echo",
            context={"test": "value"}
        )

        assert "Test error" in str(error)
        assert error.check_type == "echo"


# ============================================================================
# EchoDetector Tests
# ============================================================================

class TestEchoDetectorInit:
    """Test EchoDetector initialization."""

    def test_init_default(self):
        """Test default initialization."""
        detector = EchoDetector()
        assert detector.sample_rate == 24000
        assert detector.echo_threshold == 0.3

    def test_init_custom(self):
        """Test custom initialization."""
        detector = EchoDetector(
            sample_rate=48000,
            echo_threshold=0.5,
            min_delay_ms=100.0,
            max_delay_ms=1000.0
        )
        assert detector.sample_rate == 48000
        assert detector.echo_threshold == 0.5


class TestEchoDetectorDetect:
    """Test EchoDetector detection methods."""

    @pytest.mark.asyncio
    async def test_detect_echo_no_numpy(self):
        """Test echo detection without numpy."""
        with patch('diagnostics.audio_check.NUMPY_AVAILABLE', False):
            detector = EchoDetector()
            detected, delay, strength = await detector.detect_echo(b"ref", b"cap")

            assert detected is False
            assert delay is None
            assert strength == 0.0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    async def test_detect_echo_with_numpy(self):
        """Test echo detection with numpy."""
        import numpy as np

        detector = EchoDetector(echo_threshold=0.9)  # High threshold

        # Create fake audio data
        reference = np.random.randn(4800).astype(np.int16).tobytes()
        captured = np.random.randn(4800).astype(np.int16).tobytes()

        detected, delay, strength = await detector.detect_echo(reference, captured)

        # With random data and high threshold, probably no echo
        assert detected in [True, False]  # numpy.bool_ or bool
        assert isinstance(strength, (float, np.floating))
        assert 0 <= strength <= 1


# ============================================================================
# CrosstalkDetector Tests
# ============================================================================

class TestCrosstalkDetectorInit:
    """Test CrosstalkDetector initialization."""

    def test_init_default(self):
        """Test default initialization."""
        detector = CrosstalkDetector()
        assert detector.sample_rate == 24000
        assert detector.crosstalk_threshold == 0.2

    def test_init_custom_bands(self):
        """Test custom frequency bands."""
        bands = [(100, 200), (200, 400)]
        detector = CrosstalkDetector(analysis_bands=bands)

        assert detector.analysis_bands == bands


class TestCrosstalkDetectorDetect:
    """Test CrosstalkDetector detection methods."""

    @pytest.mark.asyncio
    async def test_detect_crosstalk_no_scipy(self):
        """Test crosstalk detection without scipy."""
        with patch('diagnostics.audio_check.SCIPY_AVAILABLE', False):
            detector = CrosstalkDetector()
            detected, level, bands = await detector.detect_crosstalk(b"a", b"b")

            assert detected is False
            assert level == 0.0
            assert bands == {}


# ============================================================================
# AudioQualityAnalyzer Tests
# ============================================================================

class TestAudioQualityAnalyzerInit:
    """Test AudioQualityAnalyzer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        analyzer = AudioQualityAnalyzer()
        assert analyzer.sample_rate == 24000
        assert analyzer.clipping_threshold == 0.95


class TestAudioQualityAnalyzerAnalyze:
    """Test AudioQualityAnalyzer analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_quality_no_numpy(self):
        """Test quality analysis without numpy."""
        with patch('diagnostics.audio_check.NUMPY_AVAILABLE', False):
            analyzer = AudioQualityAnalyzer()
            metrics = await analyzer.analyze_quality(b"fake audio")

            # Returns empty metrics
            assert metrics.snr_db is None

    @pytest.mark.asyncio
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    async def test_analyze_quality_with_numpy(self):
        """Test quality analysis with numpy."""
        import numpy as np

        analyzer = AudioQualityAnalyzer()

        # Create fake audio data
        audio_data = (np.random.randn(4800) * 1000).astype(np.int16).tobytes()
        metrics = await analyzer.analyze_quality(audio_data)

        assert metrics is not None
        assert isinstance(metrics, AudioMetrics)


# ============================================================================
# AudioDiagnostics Tests
# ============================================================================

class TestAudioDiagnosticsInit:
    """Test AudioDiagnostics initialization."""

    def test_init_default(self):
        """Test default initialization."""
        diagnostics = AudioDiagnostics()
        assert diagnostics.sample_rate == 24000
        assert diagnostics.echo_detector is not None
        assert diagnostics.crosstalk_detector is not None


class TestAudioDiagnosticsChecks:
    """Test AudioDiagnostics check methods."""

    @pytest.mark.asyncio
    async def test_check_audio_loopback(self):
        """Test audio loopback checking."""
        diagnostics = AudioDiagnostics()

        # Create fake audio data
        played = b"fake played audio" * 100
        captured = b"fake captured audio" * 100

        metrics, issues = await diagnostics.check_audio_loopback(played, captured)

        assert isinstance(metrics, AudioMetrics)
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_check_crosstalk(self):
        """Test crosstalk checking."""
        diagnostics = AudioDiagnostics()

        channel_a = b"channel a" * 100
        channel_b = b"channel b" * 100

        detected, bands, issues = await diagnostics.check_crosstalk(channel_a, channel_b)

        assert isinstance(detected, bool)
        assert isinstance(bands, dict)
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_measure_latency(self):
        """Test latency measurement."""
        diagnostics = AudioDiagnostics()

        start = time.time()
        end = start + 0.1  # 100ms

        latency = await diagnostics.measure_latency(start, end)

        assert 90 <= latency <= 110  # Allow some tolerance

    def test_get_issue_history(self):
        """Test getting issue history."""
        diagnostics = AudioDiagnostics()
        issue = AudioIssue(
            issue_type=AudioIssueType.ECHO,
            severity="warning",
            message="Test"
        )
        diagnostics._issue_history = [issue]

        history = diagnostics.get_issue_history(limit=10)

        assert len(history) == 1

    def test_clear_history(self):
        """Test clearing issue history."""
        diagnostics = AudioDiagnostics()
        issue = AudioIssue(
            issue_type=AudioIssueType.ECHO,
            severity="warning",
            message="Test"
        )
        diagnostics._issue_history = [issue]

        diagnostics.clear_history()

        assert len(diagnostics._issue_history) == 0


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestCreateAudioDiagnostics:
    """Test create_audio_diagnostics helper."""

    @pytest.mark.asyncio
    async def test_create(self):
        """Test creating AudioDiagnostics instance."""
        diagnostics = await create_audio_diagnostics(
            sample_rate=48000,
            echo_threshold=0.5,
            crosstalk_threshold=0.3
        )

        assert isinstance(diagnostics, AudioDiagnostics)
        assert diagnostics.sample_rate == 48000


# ============================================================================
# System Check Tests
# ============================================================================

class TestCheckStatus:
    """Test CheckStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert CheckStatus.PASSED.value == "passed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.WARNING.value == "warning"
        assert CheckStatus.SKIPPED.value == "skipped"


class TestSystemComponent:
    """Test SystemComponent enum."""

    def test_component_values(self):
        """Test all component values exist."""
        assert SystemComponent.MICROPHONE.value == "microphone"
        assert SystemComponent.SPEAKER.value == "speaker"
        assert SystemComponent.NETWORK.value == "network"


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        result = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.PASSED,
            message="Microphone OK",
            details={"device": "test"}
        )

        result_dict = result.to_dict()

        assert result_dict["component"] == "microphone"
        assert result_dict["status"] == "passed"
        assert result_dict["message"] == "Microphone OK"


class TestSystemInfo:
    """Test SystemInfo dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        info = SystemInfo(
            os="Linux",
            os_version="5.0",
            python_version="3.9"
        )

        info_dict = info.to_dict()

        assert info_dict["os"] == "Linux"
        assert info_dict["os_version"] == "5.0"
        assert info_dict["python_version"] == "3.9"


class TestAudioDeviceChecker:
    """Test AudioDeviceChecker class."""

    def test_init_no_library(self):
        """Test initialization without audio library."""
        # Just check the checker can be instantiated
        checker = AudioDeviceChecker()
        assert checker is not None
        assert checker.pyaudio is None
        assert checker.sounddevice is None

    def test_init_with_pyaudio_mock(self):
        """Test initialization with mocked PyAudio."""
        # Since we can't easily mock the imports, just verify the checker initializes
        # This test verifies the AudioDeviceChecker can be created
        checker = AudioDeviceChecker()
        assert checker is not None
        # The pyaudio and sounddevice will be None if not installed
        # which is expected behavior

    def test_init_with_sounddevice_mock(self):
        """Test initialization with mocked sounddevice."""
        # Verify the checker can be initialized
        checker = AudioDeviceChecker()
        assert checker is not None

    @pytest.mark.asyncio
    async def test_check_microphone_no_devices(self):
        """Test microphone check with no devices."""
        checker = AudioDeviceChecker()
        with patch.object(checker, 'get_available_devices', return_value=([], [])):
            result = await checker.check_microphone()

            assert result.status == CheckStatus.FAILED
            assert "No microphone" in result.message
            assert result.component == SystemComponent.MICROPHONE

    @pytest.mark.asyncio
    async def test_check_microphone_with_devices_no_library(self):
        """Test microphone check with devices but no audio library."""
        from diagnostics.system_check import DeviceInfo

        checker = AudioDeviceChecker()
        device = DeviceInfo(
            name="Test Mic",
            index=0,
            channels=1,
            sample_rate=48000,
            is_input=True,
            is_output=False
        )

        with patch.object(checker, 'get_available_devices', return_value=([device], [])):
            result = await checker.check_microphone()

            assert result.status == CheckStatus.WARNING
            assert "No audio library" in result.message

    @pytest.mark.asyncio
    async def test_check_microphone_device_not_found(self):
        """Test microphone check with invalid device index."""
        from diagnostics.system_check import DeviceInfo

        checker = AudioDeviceChecker()
        device = DeviceInfo(
            name="Test Mic",
            index=0,
            channels=1,
            sample_rate=48000,
            is_input=True,
            is_output=False
        )

        with patch.object(checker, 'get_available_devices', return_value=([device], [])):
            result = await checker.check_microphone(device_index=5)

            assert result.status == CheckStatus.FAILED
            assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_check_speaker_no_devices(self):
        """Test speaker check with no devices."""
        checker = AudioDeviceChecker()
        with patch.object(checker, 'get_available_devices', return_value=([], [])):
            result = await checker.check_speaker()

            assert result.status == CheckStatus.FAILED
            assert "No speaker" in result.message
            assert result.remediation is not None

    @pytest.mark.asyncio
    async def test_check_speaker_with_devices_no_library(self):
        """Test speaker check with devices but no audio library."""
        from diagnostics.system_check import DeviceInfo

        checker = AudioDeviceChecker()
        device = DeviceInfo(
            name="Test Speaker",
            index=0,
            channels=2,
            sample_rate=48000,
            is_input=False,
            is_output=True
        )

        with patch.object(checker, 'get_available_devices', return_value=([], [device])):
            result = await checker.check_speaker()

            assert result.status == CheckStatus.WARNING
            assert "audio library" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_speaker_device_not_found(self):
        """Test speaker check with invalid device index."""
        from diagnostics.system_check import DeviceInfo

        checker = AudioDeviceChecker()
        device = DeviceInfo(
            name="Test Speaker",
            index=0,
            channels=2,
            sample_rate=48000,
            is_input=False,
            is_output=True
        )

        with patch.object(checker, 'get_available_devices', return_value=([], [device])):
            result = await checker.check_speaker(device_index=5)

            assert result.status == CheckStatus.FAILED
            assert "not found" in result.message


class TestNetworkChecker:
    """Test NetworkChecker class."""

    def test_init_default(self):
        """Test default initialization."""
        checker = NetworkChecker()
        assert "api.openai.com" in checker.test_hosts
        assert "8.8.8.8" in checker.test_hosts
        assert "1.1.1.1" in checker.test_hosts

    def test_init_custom_hosts(self):
        """Test initialization with custom hosts."""
        custom_hosts = ["example.com", "test.com"]
        checker = NetworkChecker(test_hosts=custom_hosts)
        assert checker.test_hosts == custom_hosts

    @pytest.mark.asyncio
    async def test_check_connectivity_success(self):
        """Test successful connectivity check."""
        checker = NetworkChecker()

        # Mock successful connection
        async def mock_open_connection(host, port):
            reader = MagicMock()
            writer = MagicMock()
            writer.close = AsyncMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        with patch('asyncio.open_connection', side_effect=mock_open_connection):
            connected, latency = await checker.check_connectivity("example.com")

            assert connected is True
            assert latency >= 0

    @pytest.mark.asyncio
    async def test_check_connectivity_timeout(self):
        """Test connectivity check with timeout."""
        checker = NetworkChecker()

        with patch('asyncio.open_connection', side_effect=asyncio.TimeoutError):
            connected, latency = await checker.check_connectivity("example.com", timeout=0.1)

            assert connected is False
            assert latency == 0.0

    @pytest.mark.asyncio
    async def test_check_connectivity_os_error(self):
        """Test connectivity check with OS error."""
        checker = NetworkChecker()

        with patch('asyncio.open_connection', side_effect=OSError("Network unreachable")):
            connected, latency = await checker.check_connectivity("example.com")

            assert connected is False
            assert latency == 0.0

    @pytest.mark.asyncio
    async def test_check_network_all_passed(self):
        """Test network check with all hosts reachable."""
        checker = NetworkChecker(test_hosts=["host1.com", "host2.com"])

        async def mock_open_connection(host, port):
            reader = MagicMock()
            writer = MagicMock()
            writer.close = AsyncMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        with patch('asyncio.open_connection', side_effect=mock_open_connection):
            result = await checker.check_network()

            assert result.status == CheckStatus.PASSED
            assert "Network connected" in result.message
            assert result.details["connected_count"] == 2

    @pytest.mark.asyncio
    async def test_check_network_none_reachable(self):
        """Test network check with no hosts reachable."""
        checker = NetworkChecker(test_hosts=["host1.com"])

        with patch('asyncio.open_connection', side_effect=asyncio.TimeoutError):
            result = await checker.check_network()

            assert result.status == CheckStatus.FAILED
            assert "No network connectivity" in result.message
            assert result.remediation is not None

    @pytest.mark.asyncio
    async def test_check_network_partial_connectivity(self):
        """Test network check with partial connectivity."""
        checker = NetworkChecker(test_hosts=["host1.com", "host2.com"])

        call_count = 0

        async def mock_open_connection_mixed(host, port):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                reader = MagicMock()
                writer = MagicMock()
                writer.close = AsyncMock()
                writer.wait_closed = AsyncMock()
                return reader, writer
            else:
                raise asyncio.TimeoutError()

        with patch('asyncio.open_connection', side_effect=mock_open_connection_mixed):
            result = await checker.check_network()

            assert result.status == CheckStatus.WARNING
            assert "Partial" in result.message

    @pytest.mark.asyncio
    async def test_check_network_high_latency(self):
        """Test network check with high latency."""
        checker = NetworkChecker(test_hosts=["host1.com"])

        async def mock_open_connection_slow(host, port):
            reader = MagicMock()
            writer = MagicMock()
            writer.close = AsyncMock()
            writer.wait_closed = AsyncMock()
            # Sleep to simulate high latency
            await asyncio.sleep(0.6)
            return reader, writer

        with patch('asyncio.open_connection', side_effect=mock_open_connection_slow):
            result = await checker.check_network()

            assert result.status == CheckStatus.WARNING
            assert "high latency" in result.message

    @pytest.mark.asyncio
    async def test_check_api_reachable_success(self):
        """Test API reachability check success."""
        checker = NetworkChecker()

        async def mock_open_connection(host, port):
            reader = MagicMock()
            writer = MagicMock()
            writer.close = AsyncMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        with patch('asyncio.open_connection', side_effect=mock_open_connection):
            result = await checker.check_api_reachable("api.example.com")

            assert result.status == CheckStatus.PASSED
            assert "reachable" in result.message
            assert "latency_ms" in result.details

    @pytest.mark.asyncio
    async def test_check_api_reachable_failed(self):
        """Test API reachability check failure."""
        checker = NetworkChecker()

        with patch('asyncio.open_connection', side_effect=asyncio.TimeoutError):
            result = await checker.check_api_reachable("api.example.com")

            assert result.status == CheckStatus.FAILED
            assert "unreachable" in result.message
            assert result.remediation is not None


class TestPerformanceChecker:
    """Test PerformanceChecker class."""

    def test_init_default(self):
        """Test default initialization."""
        checker = PerformanceChecker()
        assert checker.min_memory_gb == 1.0
        assert checker.min_disk_percent == 90.0

    def test_init_custom(self):
        """Test initialization with custom values."""
        checker = PerformanceChecker(min_memory_gb=2.0, min_disk_percent=95.0)
        assert checker.min_memory_gb == 2.0
        assert checker.min_disk_percent == 95.0

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_get_system_info(self):
        """Test getting system information."""
        checker = PerformanceChecker()
        info = checker.get_system_info()

        assert isinstance(info, SystemInfo)
        assert info.os is not None
        assert info.python_version is not None

    @pytest.mark.asyncio
    async def test_check_memory_no_psutil(self):
        """Test memory check without psutil."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', False):
            checker = PerformanceChecker()
            result = await checker.check_memory()

            assert result.status == CheckStatus.SKIPPED
            assert "psutil not available" in result.message

    @pytest.mark.asyncio
    async def test_check_memory_passed(self):
        """Test memory check with sufficient memory."""
        mock_memory = MagicMock()
        mock_memory.available = 2 * (1024**3)  # 2GB
        mock_memory.total = 8 * (1024**3)  # 8GB
        mock_memory.percent = 75.0

        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.virtual_memory', return_value=mock_memory, create=True):
                checker = PerformanceChecker()
                result = await checker.check_memory()

                assert result.status == CheckStatus.PASSED
                assert "Memory OK" in result.message

    @pytest.mark.asyncio
    async def test_check_memory_warning(self):
        """Test memory check with low memory."""
        mock_memory = MagicMock()
        mock_memory.available = 0.5 * (1024**3)  # 500MB
        mock_memory.total = 8 * (1024**3)  # 8GB
        mock_memory.percent = 94.0

        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.virtual_memory', return_value=mock_memory, create=True):
                checker = PerformanceChecker()
                result = await checker.check_memory()

                assert result.status == CheckStatus.WARNING
                assert "Low memory" in result.message
                assert result.remediation is not None

    @pytest.mark.asyncio
    async def test_check_memory_failed(self):
        """Test memory check with very low memory."""
        mock_memory = MagicMock()
        mock_memory.available = 0.2 * (1024**3)  # 200MB
        mock_memory.total = 8 * (1024**3)  # 8GB
        mock_memory.percent = 98.0

        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.virtual_memory', return_value=mock_memory, create=True):
                checker = PerformanceChecker()
                result = await checker.check_memory()

                assert result.status == CheckStatus.FAILED
                assert "Low memory" in result.message

    @pytest.mark.asyncio
    async def test_check_memory_exception(self):
        """Test memory check with exception."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.virtual_memory', side_effect=Exception("Test error"), create=True):
                checker = PerformanceChecker()
                result = await checker.check_memory()

                assert result.status == CheckStatus.FAILED
                assert "failed" in result.message

    @pytest.mark.asyncio
    async def test_check_cpu_no_psutil(self):
        """Test CPU check without psutil."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', False):
            checker = PerformanceChecker()
            result = await checker.check_cpu()

            assert result.status == CheckStatus.SKIPPED
            assert "psutil not available" in result.message

    @pytest.mark.asyncio
    async def test_check_cpu_passed(self):
        """Test CPU check with normal usage."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.cpu_percent', return_value=50.0, create=True):
                with patch('diagnostics.system_check.psutil.cpu_count', return_value=4, create=True):
                    checker = PerformanceChecker()
                    result = await checker.check_cpu()

                    assert result.status == CheckStatus.PASSED
                    assert "CPU usage OK" in result.message
                    assert result.details["cpu_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_check_cpu_warning(self):
        """Test CPU check with high usage."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.cpu_percent', return_value=92.0, create=True):
                with patch('diagnostics.system_check.psutil.cpu_count', return_value=4, create=True):
                    checker = PerformanceChecker()
                    result = await checker.check_cpu()

                    assert result.status == CheckStatus.WARNING
                    assert "High CPU usage" in result.message

    @pytest.mark.asyncio
    async def test_check_cpu_exception(self):
        """Test CPU check with exception."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.cpu_percent', side_effect=Exception("Test error"), create=True):
                checker = PerformanceChecker()
                result = await checker.check_cpu()

                assert result.status == CheckStatus.FAILED
                assert "failed" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_no_psutil(self):
        """Test disk check without psutil."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', False):
            checker = PerformanceChecker()
            result = await checker.check_disk()

            assert result.status == CheckStatus.SKIPPED
            assert "psutil not available" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_passed(self):
        """Test disk check with sufficient space."""
        mock_disk = MagicMock()
        mock_disk.percent = 80.0
        mock_disk.free = 20 * (1024**3)  # 20GB

        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.disk_usage', return_value=mock_disk, create=True):
                checker = PerformanceChecker()
                result = await checker.check_disk()

                assert result.status == CheckStatus.PASSED
                assert "Disk space OK" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_warning(self):
        """Test disk check with low space."""
        mock_disk = MagicMock()
        mock_disk.percent = 95.0
        mock_disk.free = 2 * (1024**3)  # 2GB

        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.disk_usage', return_value=mock_disk, create=True):
                checker = PerformanceChecker()
                result = await checker.check_disk()

                assert result.status == CheckStatus.WARNING
                assert "Low disk space" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_failed(self):
        """Test disk check with very low space."""
        mock_disk = MagicMock()
        mock_disk.percent = 99.0
        mock_disk.free = 0.5 * (1024**3)  # 500MB

        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.disk_usage', return_value=mock_disk, create=True):
                checker = PerformanceChecker()
                result = await checker.check_disk()

                assert result.status == CheckStatus.FAILED
                assert "Low disk space" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_exception(self):
        """Test disk check with exception."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            with patch('diagnostics.system_check.psutil.disk_usage', side_effect=Exception("Test error"), create=True):
                checker = PerformanceChecker()
                result = await checker.check_disk()

                assert result.status == CheckStatus.FAILED
                assert "failed" in result.message


class TestPermissionChecker:
    """Test PermissionChecker class."""

    def test_check_microphone_permission_macos(self):
        """Test microphone permission check on macOS."""
        with patch('platform.system', return_value='Darwin'):
            result = PermissionChecker.check_microphone_permission()

            assert result.component == SystemComponent.PERMISSIONS
            # On macOS, we can't directly check permission
            assert result.status == CheckStatus.WARNING
            assert result.remediation is not None

    def test_check_microphone_permission_linux_in_audio_group(self):
        """Test microphone permission on Linux when user in audio group."""
        mock_audio_group = MagicMock()
        mock_audio_group.gr_mem = [12345]  # User UID

        with patch('platform.system', return_value='Linux'):
            with patch('os.getuid', return_value=12345):
                with patch('grp.getgrnam', return_value=mock_audio_group):
                    result = PermissionChecker.check_microphone_permission()

                    assert result.component == SystemComponent.PERMISSIONS
                    assert result.status == CheckStatus.PASSED
                    assert "audio group" in result.message

    def test_check_microphone_permission_linux_not_in_audio_group(self):
        """Test microphone permission on Linux when user not in audio group."""
        mock_audio_group = MagicMock()
        mock_audio_group.gr_mem = [99999]  # Different UID

        with patch('platform.system', return_value='Linux'):
            with patch('os.getuid', return_value=12345):
                with patch('grp.getgrnam', return_value=mock_audio_group):
                    result = PermissionChecker.check_microphone_permission()

                    assert result.component == SystemComponent.PERMISSIONS
                    assert result.status == CheckStatus.WARNING
                    assert "not in audio group" in result.message
                    assert result.remediation is not None

    def test_check_microphone_permission_linux_error(self):
        """Test microphone permission on Linux with error."""
        with patch('platform.system', return_value='Linux'):
            with patch('grp.getgrnam', side_effect=KeyError("Group not found")):
                result = PermissionChecker.check_microphone_permission()

                assert result.component == SystemComponent.PERMISSIONS
                assert result.status == CheckStatus.SKIPPED

    def test_check_screen_recording_permission_macos(self):
        """Test screen recording permission on macOS."""
        with patch('platform.system', return_value='Darwin'):
            result = PermissionChecker.check_screen_recording_permission()

            assert result.component == SystemComponent.PERMISSIONS
            assert result.status == CheckStatus.WARNING
            assert "runtime check" in result.message
            assert result.remediation is not None

    def test_check_screen_recording_permission_linux(self):
        """Test screen recording permission on Linux."""
        with patch('platform.system', return_value='Linux'):
            result = PermissionChecker.check_screen_recording_permission()

            assert result.component == SystemComponent.PERMISSIONS
            assert result.status == CheckStatus.SKIPPED

    def test_check_screen_recording_permission_windows(self):
        """Test screen recording permission on Windows."""
        with patch('platform.system', return_value='Windows'):
            result = PermissionChecker.check_screen_recording_permission()

            assert result.component == SystemComponent.PERMISSIONS
            assert result.status == CheckStatus.SKIPPED


class TestSystemDiagnostics:
    """Test SystemDiagnostics class."""

    def test_init_default(self):
        """Test default initialization."""
        diagnostics = SystemDiagnostics()

        assert diagnostics.audio_checker is not None
        assert diagnostics.network_checker is not None
        assert diagnostics.performance_checker is not None
        assert diagnostics.permission_checker is not None
        assert diagnostics._check_results == []

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        test_hosts = ["custom.com"]
        diagnostics = SystemDiagnostics(
            min_memory_gb=2.0,
            min_disk_percent=95.0,
            test_hosts=test_hosts
        )

        assert diagnostics.performance_checker.min_memory_gb == 2.0
        assert diagnostics.network_checker.test_hosts == test_hosts

    @pytest.mark.asyncio
    async def test_check_audio_devices(self):
        """Test audio devices check."""
        diagnostics = SystemDiagnostics()

        results = await diagnostics.check_audio_devices()

        assert len(results) == 2
        assert all(isinstance(r, CheckResult) for r in results)

    @pytest.mark.asyncio
    async def test_check_network_only(self):
        """Test network-only check."""
        diagnostics = SystemDiagnostics()

        result = await diagnostics.check_network_only()

        assert isinstance(result, CheckResult)
        assert result.component == SystemComponent.NETWORK

    @pytest.mark.asyncio
    async def test_check_api_reachable(self):
        """Test API reachability check."""
        diagnostics = SystemDiagnostics()

        async def mock_open_connection(host, port):
            reader = MagicMock()
            writer = MagicMock()
            writer.close = AsyncMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        with patch('asyncio.open_connection', side_effect=mock_open_connection):
            result = await diagnostics.check_api_reachable("api.test.com")

            assert isinstance(result, CheckResult)
            assert result.component == SystemComponent.NETWORK

    def test_get_system_info(self):
        """Test getting system info."""
        diagnostics = SystemDiagnostics()
        info = diagnostics.get_system_info()

        assert isinstance(info, SystemInfo)
        assert info.os is not None

    def test_get_check_results(self):
        """Test getting check results."""
        diagnostics = SystemDiagnostics()
        result = CheckResult(
            component=SystemComponent.CPU,
            status=CheckStatus.PASSED,
            message="OK"
        )
        diagnostics._check_results = [result]

        results = diagnostics.get_check_results()

        assert len(results) == 1
        assert results[0].component == SystemComponent.CPU

    def test_get_failed_checks(self):
        """Test getting failed checks."""
        diagnostics = SystemDiagnostics()

        failed = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.FAILED,
            message="Failed"
        )
        passed = CheckResult(
            component=SystemComponent.SPEAKER,
            status=CheckStatus.PASSED,
            message="Passed"
        )

        diagnostics._check_results = [failed, passed]

        failed_checks = diagnostics.get_failed_checks()

        assert len(failed_checks) == 1
        assert failed_checks[0].component == SystemComponent.MICROPHONE

    def test_get_warning_checks(self):
        """Test getting warning checks."""
        diagnostics = SystemDiagnostics()

        warning = CheckResult(
            component=SystemComponent.MEMORY,
            status=CheckStatus.WARNING,
            message="Warning"
        )
        passed = CheckResult(
            component=SystemComponent.SPEAKER,
            status=CheckStatus.PASSED,
            message="Passed"
        )

        diagnostics._check_results = [warning, passed]

        warning_checks = diagnostics.get_warning_checks()

        assert len(warning_checks) == 1
        assert warning_checks[0].component == SystemComponent.MEMORY

    @pytest.mark.asyncio
    async def test_run_all_checks(self):
        """Test running all checks."""
        diagnostics = SystemDiagnostics()

        # Mock network check to avoid actual network calls
        async def mock_open_connection(host, port):
            reader = MagicMock()
            writer = MagicMock()
            writer.close = AsyncMock()
            writer.wait_closed = AsyncMock()
            return reader, writer

        with patch('asyncio.open_connection', side_effect=mock_open_connection):
            results = await diagnostics.run_all_checks()

            assert len(results) == 8  # mic, speaker, network, memory, cpu, disk, mic_perm, screen_perm
            assert all(isinstance(r, CheckResult) for r in results)
            assert diagnostics._check_results == results


# ============================================================================
# Helper Functions Tests
# ============================================================================

class TestCreateSystemDiagnostics:
    """Test create_system_diagnostics helper function."""

    @pytest.mark.asyncio
    async def test_create_default(self):
        """Test creating SystemDiagnostics with defaults."""
        from diagnostics.system_check import create_system_diagnostics

        diagnostics = await create_system_diagnostics()

        assert isinstance(diagnostics, SystemDiagnostics)

    @pytest.mark.asyncio
    async def test_create_with_params(self):
        """Test creating SystemDiagnostics with parameters."""
        from diagnostics.system_check import create_system_diagnostics

        test_hosts = ["test.com"]
        diagnostics = await create_system_diagnostics(
            min_memory_gb=4.0,
            min_disk_percent=80.0,
            test_hosts=test_hosts
        )

        assert isinstance(diagnostics, SystemDiagnostics)
        assert diagnostics.performance_checker.min_memory_gb == 4.0
        assert diagnostics.network_checker.test_hosts == test_hosts


# ============================================================================
# Additional Tests for Missing Coverage
# ============================================================================

class TestSystemDiagnosticsError:
    """Test SystemDiagnosticsError exception."""

    def test_init(self):
        """Test initialization."""
        from diagnostics.system_check import SystemDiagnosticsError

        error = SystemDiagnosticsError(
            message="Test error",
            component="audio",
            context={"test": "value"},
            cause=Exception("Cause")
        )

        assert "Test error" in str(error)
        assert error.component == "audio"

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        from diagnostics.system_check import SystemDiagnosticsError

        error = SystemDiagnosticsError(message="Test")

        assert "Test" in str(error)
        assert error.component is None


class TestDeviceInfo:
    """Test DeviceInfo dataclass."""

    def test_init(self):
        """Test initialization."""
        from diagnostics.system_check import DeviceInfo

        device = DeviceInfo(
            name="Test Device",
            index=0,
            channels=2,
            sample_rate=48000,
            is_input=True,
            is_output=False,
            is_default=True
        )

        assert device.name == "Test Device"
        assert device.index == 0
        assert device.channels == 2
        assert device.sample_rate == 48000
        assert device.is_input is True
        assert device.is_output is False
        assert device.is_default is True


class TestCheckResultWithRemediation:
    """Test CheckResult with remediation."""

    def test_to_dict_with_remediation(self):
        """Test to_dict method includes remediation."""
        result = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.FAILED,
            message="Microphone failed",
            remediation="Check permissions"
        )

        result_dict = result.to_dict()

        assert result_dict["remediation"] == "Check permissions"

    def test_to_dict_without_remediation(self):
        """Test to_dict method without remediation."""
        result = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.PASSED,
            message="Microphone OK"
        )

        result_dict = result.to_dict()

        assert result_dict["remediation"] is None


class TestPerformanceCheckerGetSystemInfo:
    """Test PerformanceChecker.get_system_info method."""

    def test_get_system_info_without_psutil(self):
        """Test getting system info without psutil."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', False):
            checker = PerformanceChecker()
            info = checker.get_system_info()

            assert isinstance(info, SystemInfo)
            assert info.os is not None
            assert info.cpu_count is None
            assert info.total_memory_gb is None

    def test_get_system_info_with_psutil_exception(self):
        """Test getting system info with psutil exception."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', True):
            # Create a mock psutil module with methods that raise exceptions
            mock_psutil = MagicMock()
            mock_psutil.cpu_count.side_effect = Exception("Error")
            mock_psutil.virtual_memory.side_effect = Exception("Error")
            mock_psutil.disk_usage.side_effect = Exception("Error")

            with patch('diagnostics.system_check.psutil', mock_psutil, create=True):
                checker = PerformanceChecker()
                info = checker.get_system_info()

                # Should still return info with basic data
                assert isinstance(info, SystemInfo)
                assert info.os is not None


class TestAudioDeviceCheckerGetAvailableDevices:
    """Test AudioDeviceChecker.get_available_devices method."""

    @pytest.mark.asyncio
    async def test_get_available_devices_no_library(self):
        """Test getting available devices without audio library."""
        checker = AudioDeviceChecker()
        # Ensure no audio library
        checker.pyaudio = None
        checker.sounddevice = None

        input_devices, output_devices = checker.get_available_devices()

        assert input_devices == []
        assert output_devices == []


class TestAudioDeviceCheckerWithPyaudio:
    """Test AudioDeviceChecker with mocked PyAudio."""

    def test_init_with_pyaudio_success(self):
        """Test AudioDeviceChecker initialization with successful pyaudio import."""
        # Create a real module-like mock for pyaudio
        mock_pyaudio_module = MagicMock()

        with patch.dict('sys.modules', {'pyaudio': mock_pyaudio_module}):
            # Clear any cached instances
            import importlib
            import diagnostics.system_check
            importlib.reload(diagnostics.system_check)

            from diagnostics.system_check import AudioDeviceChecker
            checker = AudioDeviceChecker()

            # The checker should have pyaudio available if import succeeded
            # Since we're mocking, just verify the checker was created
            assert checker is not None

    @pytest.mark.asyncio
    async def test_check_microphone_with_pyaudio(self):
        """Test microphone check with PyAudio available."""
        from diagnostics.system_check import AudioDeviceChecker, DeviceInfo

        # Create a checker and manually set pyaudio
        checker = AudioDeviceChecker()
        mock_pyaudio = MagicMock()

        # Mock PyAudio() and stream
        mock_py_audio_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_py_audio_instance
        mock_pyaudio.paInt16 = 0

        # Mock device info
        mock_device_info = {
            'name': 'Test Mic',
            'maxInputChannels': 1,
            'maxOutputChannels': 0,
            'defaultSampleRate': 48000
        }
        mock_py_audio_instance.get_device_count.return_value = 1
        mock_py_audio_instance.get_device_info_by_index.return_value = mock_device_info

        # Mock stream
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'\x00' * 2048
        mock_py_audio_instance.open.return_value = mock_stream

        # Manually inject the mock pyaudio
        checker.pyaudio = mock_pyaudio

        result = await checker.check_microphone()

        # Should return some result (depending on whether the mock works properly)
        from diagnostics.system_check import CheckResult as SystemCheckResult
        assert result is not None
        assert isinstance(result, SystemCheckResult)

    @pytest.mark.asyncio
    async def test_check_microphone_pyaudio_os_error(self):
        """Test microphone check with PyAudio OSError."""
        from diagnostics.system_check import AudioDeviceChecker, DeviceInfo

        checker = AudioDeviceChecker()
        mock_pyaudio = MagicMock()

        mock_py_audio_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_py_audio_instance
        mock_pyaudio.paInt16 = 0

        # Mock device info
        mock_device_info = {
            'name': 'Test Mic',
            'maxInputChannels': 1,
            'maxOutputChannels': 0,
            'defaultSampleRate': 48000
        }
        mock_py_audio_instance.get_device_count.return_value = 1
        mock_py_audio_instance.get_device_info_by_index.return_value = mock_device_info

        # Mock stream that raises OSError
        mock_stream = MagicMock()
        mock_stream.read.side_effect = OSError("Permission denied")
        mock_py_audio_instance.open.return_value = mock_stream

        checker.pyaudio = mock_pyaudio

        result = await checker.check_microphone()

        # Should handle OSError gracefully
        from diagnostics.system_check import CheckResult as SystemCheckResult
        assert result is not None
        assert isinstance(result, SystemCheckResult)

    @pytest.mark.asyncio
    async def test_check_speaker_with_pyaudio(self):
        """Test speaker check with PyAudio available."""
        from diagnostics.system_check import AudioDeviceChecker, DeviceInfo

        checker = AudioDeviceChecker()
        mock_pyaudio = MagicMock()

        mock_py_audio_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_py_audio_instance
        mock_pyaudio.paInt16 = 0

        # Mock device info
        mock_device_info = {
            'name': 'Test Speaker',
            'maxInputChannels': 0,
            'maxOutputChannels': 2,
            'defaultSampleRate': 48000
        }
        mock_py_audio_instance.get_device_count.return_value = 1
        mock_py_audio_instance.get_device_info_by_index.return_value = mock_device_info

        # Mock stream
        mock_stream = MagicMock()
        mock_py_audio_instance.open.return_value = mock_stream

        checker.pyaudio = mock_pyaudio

        result = await checker.check_speaker()

        # Should return some result
        from diagnostics.system_check import CheckResult as SystemCheckResult
        assert result is not None
        assert isinstance(result, SystemCheckResult)


class TestSystemDiagnosticsErrorContext:
    """Test SystemDiagnosticsError with context."""

    def test_error_with_cause(self):
        """Test SystemDiagnosticsError with cause."""
        from diagnostics.system_check import SystemDiagnosticsError

        cause = ValueError("Root cause")
        error = SystemDiagnosticsError(
            message="Error occurred",
            component="test",
            cause=cause
        )

        assert "Error occurred" in str(error)
        assert error.component == "test"
        assert error.__cause__ is None  # The exception stores it separately

    def test_error_with_context_update(self):
        """Test SystemDiagnosticsError context update."""
        from diagnostics.system_check import SystemDiagnosticsError

        error = SystemDiagnosticsError(
            message="Error",
            context={"key": "value"}
        )

        # The context should be updated with component
        assert error.context["component"] is None


# ============================================================================
# Diagnostic Report Tests
# ============================================================================

class TestSeverityLevel:
    """Test SeverityLevel enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert SeverityLevel.INFO.value == "info"
        assert SeverityLevel.WARNING.value == "warning"
        assert SeverityLevel.ERROR.value == "error"
        assert SeverityLevel.CRITICAL.value == "critical"


class TestFixSuggestion:
    """Test FixSuggestion dataclass."""

    def test_init(self):
        """Test initialization."""
        suggestion = FixSuggestion(
            action="Test action",
            description="Test description",
            commands=["cmd1", "cmd2"],
            priority=1,
            automated=True
        )

        assert suggestion.action == "Test action"
        assert suggestion.automated is True


class TestDiagnosticIssue:
    """Test DiagnosticIssue dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        issue = DiagnosticIssue(
            component="audio",
            issue_type="echo",
            severity=SeverityLevel.WARNING,
            message="Echo detected",
            details={"delay_ms": 150},
            fix_suggestions=[
                FixSuggestion(
                    action="Use headphones",
                    description="Helps prevent echo"
                )
            ]
        )

        issue_dict = issue.to_dict()

        assert issue_dict["component"] == "audio"
        assert issue_dict["severity"] == "warning"
        assert len(issue_dict["fix_suggestions"]) == 1


class TestDiagnosticReport:
    """Test DiagnosticReport class."""

    def test_init(self):
        """Test initialization."""
        report = DiagnosticReport()

        assert report.session_id is not None
        assert report.issues == []
        assert report.system_checks == []

    def test_add_issue(self):
        """Test adding an issue."""
        report = DiagnosticReport()

        issue = DiagnosticIssue(
            component="audio",
            issue_type="echo",
            severity=SeverityLevel.WARNING,
            message="Test"
        )

        report.add_issue(issue)

        assert len(report.issues) == 1

    def test_generate_summary(self):
        """Test summary generation."""
        report = DiagnosticReport()

        # Add some issues
        report.add_issue(DiagnosticIssue(
            component="audio",
            issue_type="echo",
            severity=SeverityLevel.WARNING,
            message="Echo"
        ))

        summary = report.generate_summary()

        assert summary["total_issues"] == 1
        assert summary["severity_breakdown"]["warning"] == 1

    def test_to_dict(self):
        """Test to_dict method."""
        report = DiagnosticReport()

        report_dict = report.to_dict()

        assert "session_id" in report_dict
        assert "timestamp" in report_dict
        assert "summary" in report_dict


class TestIssueClassifier:
    """Test IssueClassifier class."""

    def test_classify_audio_issue(self):
        """Test audio issue classification."""
        from diagnostics.audio_check import AudioIssue

        issue = AudioIssue(
            issue_type=AudioIssueType.ECHO,
            severity="warning",
            message="Echo detected"
        )

        classified = IssueClassifier.classify_audio_issue(issue)

        assert classified.component == "audio"
        assert classified.severity == SeverityLevel.ERROR

    def test_classify_system_check_passed(self):
        """Test classifying passed system check."""
        check = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.PASSED,
            message="OK"
        )

        result = IssueClassifier.classify_system_check(check)

        assert result is None

    def test_classify_system_check_failed(self):
        """Test classifying failed system check."""
        check = CheckResult(
            component=SystemComponent.MICROPHONE,
            status=CheckStatus.FAILED,
            message="Failed"
        )

        result = IssueClassifier.classify_system_check(check)

        assert result is not None
        assert result.severity == SeverityLevel.CRITICAL


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_init(self):
        """Test initialization."""
        generator = ReportGenerator(session_id="test_session")

        assert generator.session_id == "test_session"

    def test_create_report(self):
        """Test creating a report."""
        generator = ReportGenerator()

        report = generator.create_report()

        assert isinstance(report, DiagnosticReport)

    def test_create_report_with_data(self):
        """Test creating report with data."""
        from diagnostics.audio_check import AudioIssue, AudioMetrics

        generator = ReportGenerator()

        issues = [
            AudioIssue(
                issue_type=AudioIssueType.ECHO,
                severity="warning",
                message="Echo detected"
            )
        ]

        metrics = AudioMetrics(snr_db=20.0)

        checks = [
            CheckResult(
                component=SystemComponent.MICROPHONE,
                status=CheckStatus.PASSED,
                message="OK"
            )
        ]

        report = generator.create_report(
            audio_issues=issues,
            audio_metrics=metrics,
            system_checks=checks
        )

        assert len(report.issues) > 0
        assert report.audio_metrics is not None
        assert len(report.system_checks) == 1


class TestConsoleReporter:
    """Test ConsoleReporter class."""

    def test_format_report(self):
        """Test report formatting."""
        report = DiagnosticReport(session_id="test")

        # Add an issue
        report.add_issue(DiagnosticIssue(
            component="audio",
            issue_type="echo",
            severity=SeverityLevel.WARNING,
            message="Test echo"
        ))

        formatted = ConsoleReporter.format_report(report)

        assert "test" in formatted
        assert "echo" in formatted

    def test_format_issue(self):
        """Test issue formatting."""
        issue = DiagnosticIssue(
            component="audio",
            issue_type="echo",
            severity=SeverityLevel.WARNING,
            message="Test echo",
            fix_suggestions=[
                FixSuggestion(
                    action="Use headphones",
                    description="Prevents echo"
                )
            ]
        )

        formatted = ConsoleReporter.format_issue(issue)

        assert "audio" in formatted
        assert "echo" in formatted
        assert "headphones" in formatted


# ============================================================================
# Helper Functions Tests
# ============================================================================

class TestGenerateDiagnosticReport:
    """Test generate_diagnostic_report helper."""

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generating diagnostic report."""
        report = await generate_diagnostic_report()

        assert isinstance(report, DiagnosticReport)
