"""Tests for diagnostics module."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

try:
    from diagnostics.audio_check import (
        AudioDiagnostics,
        AudioDiagnosticsError,
        AudioIssue,
        AudioIssueType,
        AudioMetrics,
        AudioQualityAnalyzer,
        CrosstalkDetector,
        EchoDetector,
        NUMPY_AVAILABLE,
        SCIPY_AVAILABLE,
        create_audio_diagnostics,
    )
except ImportError:
    pytest.skip("diagnostics module not available", allow_module_level=True)

from diagnostics.system_check import (
    AudioDeviceChecker,
    CheckResult,
    CheckStatus,
    NetworkChecker,
    PerformanceChecker,
    PermissionChecker,
    SystemComponent,
    SystemDiagnostics,
    SystemInfo,
    DeviceInfo,
    PSUTIL_AVAILABLE,
)
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

    @pytest.mark.asyncio
    async def test_check_microphone_no_devices(self):
        """Test microphone check with no devices."""
        checker = AudioDeviceChecker()
        with patch.object(checker, 'get_available_devices', return_value=([], [])):
            result = await checker.check_microphone()

            assert result.status == CheckStatus.FAILED
            assert "No microphone" in result.message


class TestNetworkChecker:
    """Test NetworkChecker class."""

    def test_init_default(self):
        """Test default initialization."""
        checker = NetworkChecker()
        assert "api.openai.com" in checker.test_hosts

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


class TestPerformanceChecker:
    """Test PerformanceChecker class."""

    def test_init_default(self):
        """Test default initialization."""
        checker = PerformanceChecker()
        assert checker.min_memory_gb == 1.0

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_get_system_info(self):
        """Test getting system information."""
        checker = PerformanceChecker()
        info = checker.get_system_info()

        assert isinstance(info, SystemInfo)
        assert info.os is not None

    @pytest.mark.asyncio
    async def test_check_memory_no_psutil(self):
        """Test memory check without psutil."""
        with patch('diagnostics.system_check.PSUTIL_AVAILABLE', False):
            checker = PerformanceChecker()
            result = await checker.check_memory()

            assert result.status == CheckStatus.SKIPPED


class TestPermissionChecker:
    """Test PermissionChecker class."""

    def test_check_microphone_permission_macos(self):
        """Test microphone permission check on macOS."""
        with patch('platform.system', return_value='Darwin'):
            result = PermissionChecker.check_microphone_permission()

            assert result.component == SystemComponent.PERMISSIONS
            # On macOS, we can't directly check permission
            assert result.status == CheckStatus.WARNING

    def test_check_screen_recording_permission_linux(self):
        """Test screen recording permission on Linux."""
        with patch('platform.system', return_value='Linux'):
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
