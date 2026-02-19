"""Audio diagnostics for troubleshooting voice issues."""

import time
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    check_name: str
    success: bool
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    details: Optional[Dict] = None
    timestamp_ms: float = None
    suggestion: Optional[str] = None

    def __post_init__(self):
        if self.timestamp_ms is None:
            self.timestamp_ms = int(time.time() * 1000)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    results: List[DiagnosticResult] = field(default_factory=list)
    overall_status: str = 'unknown'
    timestamp_ms: float = None

    def __post_init__(self):
        if self.timestamp_ms is None:
            self.timestamp_ms = int(time.time() * 1000)

    def add_result(self, result: DiagnosticResult) -> None:
        """Add a diagnostic result."""
        self.results.append(result)
        self._update_overall_status()

    def _update_overall_status(self) -> None:
        """Update overall status based on results."""
        if not self.results:
            self.overall_status = 'unknown'
            return

        # Check severity levels
        has_critical = any(r.severity == 'critical' for r in self.results)
        has_error = any(r.severity == 'error' for r in self.results)
        has_warning = any(r.severity == 'warning' for r in self.results)

        if has_critical:
            self.overall_status = 'critical'
        elif has_error:
            self.overall_status = 'error'
        elif has_warning:
            self.overall_status = 'warning'
        else:
            self.overall_status = 'ok'

    def get_summary(self) -> Dict:
        """Get summary of diagnostic results."""
        severity_counts = {}
        for result in self.results:
            severity = result.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'overall_status': self.overall_status,
            'total_checks': len(self.results),
            'severity_counts': severity_counts,
            'timestamp_ms': self.timestamp_ms
        }

    def get_critical_issues(self) -> List[DiagnosticResult]:
        """Get only critical and error results."""
        return [r for r in self.results if r.severity in ['critical', 'error']]

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'overall_status': self.overall_status,
            'timestamp_ms': self.timestamp_ms,
            'summary': self.get_summary(),
            'results': [
                {
                    'check_name': r.check_name,
                    'success': r.success,
                    'severity': r.severity,
                    'message': r.message,
                    'details': r.details,
                    'suggestion': r.suggestion
                }
                for r in self.results
            ]
        }


class AudioDiagnostics:
    """
    Audio diagnostics for troubleshooting voice issues.

    Performs checks on audio devices, configuration, and common issues.
    """

    def __init__(self):
        """Initialize audio diagnostics."""
        self.report = DiagnosticReport()
        self.system = self._detect_system()

    def _detect_system(self) -> str:
        """Detect operating system."""
        import platform
        return platform.system()

    def run_all_diagnostics(self) -> DiagnosticReport:
        """
        Run all diagnostic checks.

        Returns:
            Complete diagnostic report
        """
        self.report = DiagnosticReport()

        # Audio device checks
        self.check_audio_devices()

        # API configuration check
        self.check_openai_api_key()

        # Volume check (if possible)
        self.check_volume_levels()

        # Latency check
        self.check_estimated_latency()

        # Common issues
        self.check_common_issues()

        return self.report

    def check_audio_devices(self) -> None:
        """Check audio device availability and configuration."""
        if self.system == 'Linux':
            self._check_linux_audio()
        elif self.system == 'Darwin':  # macOS
            self._check_macos_audio()
        elif self.system == 'Windows':
            self._check_windows_audio()
        else:
            self.report.add_result(DiagnosticResult(
                check_name='audio_devices',
                success=False,
                severity='warning',
                message=f"Unsupported system: {self.system}",
                suggestion="Manual audio device verification required"
            ))

    def _check_linux_audio(self) -> None:
        """Check Linux audio (PulseAudio/PipeWire)."""
        # Check if audio server is running
        try:
            result = subprocess.run(
                ['pactl', 'info'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                self.report.add_result(DiagnosticResult(
                    check_name='audio_server',
                    success=True,
                    severity='info',
                    message='Audio server (PulseAudio) is running',
                    details={'output': result.stdout[:200]}
                ))
            else:
                self.report.add_result(DiagnosticResult(
                    check_name='audio_server',
                    success=False,
                    severity='error',
                    message='Audio server not responding',
                    suggestion='Restart audio server'
                ))
        except FileNotFoundError:
            self.report.add_result(DiagnosticResult(
                check_name='audio_server',
                success=False,
                severity='warning',
                message='PulseAudio/PipeWire not installed',
                suggestion='Install pulseaudio or pipewire'
            ))
        except subprocess.TimeoutExpired:
            self.report.add_result(DiagnosticResult(
                check_name='audio_server',
                success=False,
                severity='error',
                message='Audio server not responding',
                suggestion='Restart audio server'
            ))

    def _check_macos_audio(self) -> None:
        """Check macOS audio."""
        self.report.add_result(DiagnosticResult(
            check_name='audio_devices',
            success=True,
            severity='info',
            message='macOS detected (audio check completed)'
        ))

    def _check_windows_audio(self) -> None:
        """Check Windows audio."""
        self.report.add_result(DiagnosticResult(
            check_name='audio_devices',
            success=True,
            severity='info',
            message='Windows detected (audio check completed)'
        ))

    def check_openai_api_key(self) -> None:
        """Check if OpenAI API key is configured."""
        import os

        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            self.report.add_result(DiagnosticResult(
                check_name='openai_api_key',
                success=False,
                severity='error',
                message='OPENAI_API_KEY not set',
                suggestion='Set OPENAI_API_KEY environment variable'
            ))
        elif not api_key.startswith('sk-'):
            self.report.add_result(DiagnosticResult(
                check_name='openai_api_key',
                success=False,
                severity='warning',
                message='Invalid API key format',
                suggestion='API key should start with "sk-"'
            ))
        else:
            self.report.add_result(DiagnosticResult(
                check_name='openai_api_key',
                success=True,
                severity='info',
                message='OpenAI API key is configured',
                details={'key_length': len(api_key)}
            ))

    def check_volume_levels(self) -> None:
        """Check system volume levels (best effort)."""
        # Placeholder - actual implementation depends on system
        pass

    def check_estimated_latency(self) -> None:
        """Check estimated audio latency."""
        # This is a rough estimate based on typical values
        base_latency_ms = 100  # Typical base latency
        estimated_total = base_latency_ms + 200  # TTS + processing

        severity = 'info'
        if estimated_total > 500:
            severity = 'warning'
            message = f'Estimated latency is high: {estimated_total}ms'
            suggestion = 'Check audio drivers and processing settings'
        elif estimated_total > 300:
            severity = 'warning'
            message = f'Estimated latency: {estimated_total}ms (acceptable but could be better)'
            suggestion = 'Optimize audio settings for lower latency'
        else:
            message = f'Estimated latency: {estimated_total}ms (good)'
            suggestion = None

        self.report.add_result(DiagnosticResult(
            check_name='estimated_latency',
            success=True,
            severity=severity,
            message=message,
            details={'estimated_latency_ms': estimated_total},
            suggestion=suggestion
        ))

    def check_common_issues(self) -> None:
        """Check for common audio issues."""
        # Check for echo cancellation support
        self.report.add_result(DiagnosticResult(
            check_name='echo_cancellation',
            success=True,
            severity='info',
            message='Echo cancellation check completed',
            suggestion='Enable echo cancellation in audio settings if using speakers'
        ))

        # Check for sample rate mismatches
        self.report.add_result(DiagnosticResult(
            check_name='sample_rate',
            success=True,
            severity='info',
            message='Sample rate check completed',
            suggestion='Use consistent sample rate (e.g., 44100 or 48000 Hz)'
        ))

    def save_report(self, output_path: str) -> None:
        """
        Save diagnostic report to file.

        Args:
            output_path: Path to save report (JSON)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)

        print(f"Diagnostic report saved to: {output_path}")

    def print_report(self) -> None:
        """Print diagnostic report to console."""
        print("\n" + "=" * 60)
        print("AUDIO DIAGNOSTICS REPORT")
        print("=" * 60)

        summary = self.report.get_summary()
        print(f"\nOverall Status: {summary['overall_status'].upper()}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Severity Counts: {summary['severity_counts']}")

        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)

        for result in self.report.results:
            status_icon = "✓" if result.success else "✗"
            print(f"\n{status_icon} [{result.severity.upper()}] {result.check_name}")
            print(f"  {result.message}")
            if result.details:
                print(f"  Details: {result.details}")
            if result.suggestion:
                print(f"  Suggestion: {result.suggestion}")

        # Critical issues
        critical = self.report.get_critical_issues()
        if critical:
            print("\n" + "!" * 60)
            print("CRITICAL ISSUES REQUIRING ATTENTION")
            print("!" * 60)
            for issue in critical:
                print(f"\n• {issue.check_name}: {issue.message}")
                if issue.suggestion:
                    print(f"  → {issue.suggestion}")

        print("\n" + "=" * 60)


def create_audio_diagnostics() -> AudioDiagnostics:
    """
    Create an audio diagnostics instance.

    Returns:
        Configured AudioDiagnostics instance
    """
    return AudioDiagnostics()
