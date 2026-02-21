"""Diagnostic report generation and issue classification.

This module provides functionality for:
- Generating diagnostic reports from check results
- Classifying issues by severity
- Providing automatic fix suggestions
- Exporting reports in various formats
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from .audio_check import AudioIssue, AudioMetrics, AudioIssueType
from .system_check import CheckResult, CheckStatus, SystemComponent

logger = get_logger(__name__)


class SeverityLevel(Enum):
    """Severity levels for diagnostic issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FixSuggestion:
    """Suggested fix for a diagnostic issue."""
    action: str
    description: str
    commands: Optional[List[str]] = None
    priority: int = 0  # Lower is higher priority
    automated: bool = False  # Can this be automatically fixed?


@dataclass
class DiagnosticIssue:
    """A diagnostic issue with classification and fix suggestions."""
    component: str
    issue_type: str
    severity: SeverityLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
            "fix_suggestions": [
                {
                    "action": s.action,
                    "description": s.description,
                    "commands": s.commands,
                    "priority": s.priority,
                    "automated": s.automated
                }
                for s in self.fix_suggestions
            ]
        }


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        """Initialize diagnostic report.

        Args:
            session_id: Optional session identifier
            timestamp: Report timestamp (default: current time)
        """
        self.session_id = session_id or f"diag_{int(time.time())}"
        self.timestamp = timestamp or time.time()
        self.issues: List[DiagnosticIssue] = []
        self.audio_metrics: Optional[AudioMetrics] = None
        self.system_checks: List[CheckResult] = []
        self.summary: Dict[str, Any] = {}

    def add_issue(self, issue: DiagnosticIssue) -> None:
        """Add an issue to the report.

        Args:
            issue: DiagnosticIssue to add
        """
        self.issues.append(issue)

    def add_issues(self, issues: List[DiagnosticIssue]) -> None:
        """Add multiple issues to the report.

        Args:
            issues: List of DiagnosticIssue to add
        """
        self.issues.extend(issues)

    def set_audio_metrics(self, metrics: AudioMetrics) -> None:
        """Set audio metrics.

        Args:
            metrics: AudioMetrics object
        """
        self.audio_metrics = metrics

    def add_system_check(self, check: CheckResult) -> None:
        """Add a system check result.

        Args:
            check: CheckResult to add
        """
        self.system_checks.append(check)

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Summary dictionary
        """
        severity_counts = {level.value: 0 for level in SeverityLevel}
        component_counts: Dict[str, int] = {}

        for issue in self.issues:
            severity_counts[issue.severity.value] += 1
            component_counts[issue.component] = component_counts.get(issue.component, 0) + 1

        # Count system checks by status
        check_counts = {status.value: 0 for status in CheckStatus}
        for check in self.system_checks:
            check_counts[check.status.value] += 1

        self.summary = {
            "total_issues": len(self.issues),
            "severity_breakdown": severity_counts,
            "component_breakdown": component_counts,
            "system_checks": check_counts,
            "has_audio_data": self.audio_metrics is not None,
            "overall_status": self._calculate_overall_status()
        }

        return self.summary

    def _calculate_overall_status(self) -> str:
        """Calculate overall status based on issues.

        Returns:
            Overall status string
        """
        if any(i.severity == SeverityLevel.CRITICAL for i in self.issues):
            return "critical"
        if any(i.severity == SeverityLevel.ERROR for i in self.issues):
            return "error"
        if any(i.severity == SeverityLevel.WARNING for i in self.issues):
            return "warning"
        return "healthy"

    def get_issues_by_severity(self, severity: SeverityLevel) -> List[DiagnosticIssue]:
        """Get issues filtered by severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of DiagnosticIssue with matching severity
        """
        return [i for i in self.issues if i.severity == severity]

    def get_issues_by_component(self, component: str) -> List[DiagnosticIssue]:
        """Get issues filtered by component.

        Args:
            component: Component name to filter by

        Returns:
            List of DiagnosticIssue with matching component
        """
        return [i for i in self.issues if i.component == component]

    def get_critical_issues(self) -> List[DiagnosticIssue]:
        """Get all critical issues.

        Returns:
            List of critical DiagnosticIssue
        """
        return self.get_issues_by_severity(SeverityLevel.CRITICAL)

    def get_error_issues(self) -> List[DiagnosticIssue]:
        """Get all error-level issues.

        Returns:
            List of error-level DiagnosticIssue
        """
        return self.get_issues_by_severity(SeverityLevel.ERROR)

    def get_warning_issues(self) -> List[DiagnosticIssue]:
        """Get all warning-level issues.

        Returns:
            List of warning-level DiagnosticIssue
        """
        return self.get_issues_by_severity(SeverityLevel.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dictionary representation of report
        """
        self.generate_summary()

        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
            "audio_metrics": self.audio_metrics.to_dict() if self.audio_metrics else None,
            "system_checks": [c.to_dict() for c in self.system_checks]
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> None:
        """Save report to file.

        Args:
            path: File path to save report
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        logger.info(f"Diagnostic report saved to {path}")


class IssueClassifier:
    """Classifies diagnostic issues and provides fix suggestions."""

    # Predefined fix suggestions for common issues
    FIX_SUGGESTIONS: Dict[str, List[FixSuggestion]] = {
        "echo": [
            FixSuggestion(
                action="Use headphones",
                description="Use headphones instead of speakers to prevent audio from being picked up by the microphone",
                priority=1
            ),
            FixSuggestion(
                action="Adjust microphone position",
                description="Move microphone away from speakers or use a directional microphone",
                priority=2
            ),
            FixSuggestion(
                action="Enable echo cancellation",
                description="Enable software echo cancellation in audio settings",
                priority=3,
                automated=True
            ),
        ],
        "crosstalk": [
            FixSuggestion(
                action="Check audio routing",
                description="Ensure audio channels are properly separated in audio settings",
                priority=1
            ),
            FixSuggestion(
                action="Use different audio devices",
                description="Use separate devices for input and output if possible",
                priority=2
            ),
        ],
        "low_snr": [
            FixSuggestion(
                action="Reduce background noise",
                description="Move to a quieter environment or use noise reduction",
                priority=1
            ),
            FixSuggestion(
                action="Adjust microphone gain",
                description="Increase microphone gain or move closer to the microphone",
                priority=2
            ),
        ],
        "clipping": [
            FixSuggestion(
                action="Lower microphone gain",
                description="Reduce microphone gain in audio settings to prevent distortion",
                priority=1
            ),
            FixSuggestion(
                action="Move away from microphone",
                description="Increase distance from microphone or speak more softly",
                priority=2
            ),
        ],
        "no_microphone": [
            FixSuggestion(
                action="Connect microphone",
                description="Ensure a microphone is properly connected",
                priority=1
            ),
            FixSuggestion(
                action="Check permissions",
                description="Grant microphone access in system preferences",
                priority=2,
                automated=True
            ),
            FixSuggestion(
                action="Install audio drivers",
                description="Install or update audio drivers if needed",
                commands=["brew install portaudio", "pip install pyaudio"],
                priority=3
            ),
        ],
        "no_speaker": [
            FixSuggestion(
                action="Connect speakers",
                description="Ensure speakers or headphones are properly connected",
                priority=1
            ),
            FixSuggestion(
                action="Check volume",
                description="Verify system volume is not muted",
                priority=2
            ),
        ],
        "network_unreachable": [
            FixSuggestion(
                action="Check internet connection",
                description="Verify internet connectivity is working",
                priority=1
            ),
            FixSuggestion(
                action="Check firewall",
                description="Ensure firewall is not blocking the application",
                priority=2
            ),
            FixSuggestion(
                action="Test connection",
                description="Run network diagnostics: ping api.openai.com",
                commands=["ping -c 4 api.openai.com"],
                priority=3
            ),
        ],
        "high_latency": [
            FixSuggestion(
                action="Use wired connection",
                description="Switch from WiFi to wired ethernet for lower latency",
                priority=1
            ),
            FixSuggestion(
                action="Close bandwidth-intensive apps",
                description="Close downloads, streaming, or other high-bandwidth applications",
                priority=2
            ),
        ],
        "low_memory": [
            FixSuggestion(
                action="Close applications",
                description="Close unnecessary applications to free memory",
                priority=1
            ),
            FixSuggestion(
                action="Restart application",
                description="Restart the application to clear memory leaks",
                priority=2
            ),
        ],
        "screen_permission": [
            FixSuggestion(
                action="Grant screen recording permission",
                description="Go to System Preferences > Privacy & Security > Screen Recording and enable for this application",
                priority=1
            ),
            FixSuggestion(
                action="Restart application",
                description="Restart the application after granting permission",
                priority=2
            ),
        ],
    }

    @classmethod
    def classify_audio_issue(cls, issue: AudioIssue) -> DiagnosticIssue:
        """Classify an audio issue and add fix suggestions.

        Args:
            issue: AudioIssue to classify

        Returns:
            DiagnosticIssue with classification and suggestions
        """
        # Map issue type to severity
        severity_mapping = {
            AudioIssueType.NO_SIGNAL: SeverityLevel.WARNING,
            AudioIssueType.CLIPPING: SeverityLevel.WARNING,
            AudioIssueType.LOW_SNR: SeverityLevel.WARNING,
            AudioIssueType.HIGH_LATENCY: SeverityLevel.WARNING,
            AudioIssueType.CROSSTALK: SeverityLevel.ERROR,
            AudioIssueType.ECHO: SeverityLevel.ERROR,
        }

        severity = severity_mapping.get(
            issue.issue_type,
            SeverityLevel.INFO
        )

        # Override severity from issue if specified
        if issue.severity == "error":
            severity = SeverityLevel.ERROR
        elif issue.severity == "warning" and severity == SeverityLevel.INFO:
            severity = SeverityLevel.WARNING

        # Get fix suggestions
        suggestions_key = issue.issue_type.value
        fix_suggestions = cls.FIX_SUGGESTIONS.get(suggestions_key, [])

        return DiagnosticIssue(
            component="audio",
            issue_type=issue.issue_type.value,
            severity=severity,
            message=issue.message,
            timestamp=issue.timestamp,
            details=issue.details,
            fix_suggestions=fix_suggestions
        )

    @classmethod
    def classify_system_check(cls, check: CheckResult) -> Optional[DiagnosticIssue]:
        """Classify a system check result as an issue if needed.

        Args:
            check: CheckResult to classify

        Returns:
            DiagnosticIssue if check failed, None otherwise
        """
        if check.status == CheckStatus.PASSED or check.status == CheckStatus.SKIPPED:
            return None

        # Map component to suggestions key
        component_mapping = {
            SystemComponent.MICROPHONE: "no_microphone",
            SystemComponent.SPEAKER: "no_speaker",
            SystemComponent.NETWORK: "network_unreachable",
            SystemComponent.MEMORY: "low_memory",
            SystemComponent.PERMISSIONS: "screen_permission",
        }

        suggestions_key = component_mapping.get(check.component, "")
        fix_suggestions = cls.FIX_SUGGESTIONS.get(suggestions_key, [])

        # Determine severity
        if check.status == CheckStatus.FAILED:
            severity = SeverityLevel.ERROR
            # Critical for certain components
            if check.component in (SystemComponent.MICROPHONE, SystemComponent.NETWORK):
                severity = SeverityLevel.CRITICAL
        else:
            severity = SeverityLevel.WARNING

        return DiagnosticIssue(
            component=check.component.value,
            issue_type=f"system_check_{check.status.value}",
            severity=severity,
            message=check.message,
            timestamp=check.timestamp,
            details=check.details,
            fix_suggestions=fix_suggestions
        )

    @classmethod
    def get_severity_from_string(cls, severity_str: str) -> SeverityLevel:
        """Convert severity string to SeverityLevel enum.

        Args:
            severity_str: Severity string ("info", "warning", "error", "critical")

        Returns:
            SeverityLevel enum value
        """
        mapping = {
            "info": SeverityLevel.INFO,
            "warning": SeverityLevel.WARNING,
            "error": SeverityLevel.ERROR,
            "critical": SeverityLevel.CRITICAL,
        }
        return mapping.get(severity_str.lower(), SeverityLevel.INFO)


class ReportGenerator:
    """Generates diagnostic reports from check results."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize report generator.

        Args:
            session_id: Optional session identifier for reports
        """
        self.session_id = session_id
        self.classifier = IssueClassifier()

    def create_report(
        self,
        audio_issues: Optional[List[AudioIssue]] = None,
        audio_metrics: Optional[AudioMetrics] = None,
        system_checks: Optional[List[CheckResult]] = None
    ) -> DiagnosticReport:
        """Create a diagnostic report from check results.

        Args:
            audio_issues: List of audio issues found
            audio_metrics: Audio metrics collected
            system_checks: List of system check results

        Returns:
            DiagnosticReport object
        """
        report = DiagnosticReport(session_id=self.session_id)

        # Classify and add audio issues
        if audio_issues:
            for issue in audio_issues:
                classified = self.classifier.classify_audio_issue(issue)
                report.add_issue(classified)

        # Set audio metrics
        if audio_metrics:
            report.set_audio_metrics(audio_metrics)

        # Classify and add system checks
        if system_checks:
            for check in system_checks:
                report.add_system_check(check)
                classified = self.classifier.classify_system_check(check)
                if classified:
                    report.add_issue(classified)

        # Generate summary
        report.generate_summary()

        return report

    async def generate_and_save_report(
        self,
        audio_issues: Optional[List[AudioIssue]] = None,
        audio_metrics: Optional[AudioMetrics] = None,
        system_checks: Optional[List[CheckResult]] = None,
        output_dir: Optional[str] = None
    ) -> DiagnosticReport:
        """Generate and save a diagnostic report.

        Args:
            audio_issues: List of audio issues found
            audio_metrics: Audio metrics collected
            system_checks: List of system check results
            output_dir: Directory to save report (default: ./logs/diagnostics)

        Returns:
            DiagnosticReport object
        """
        report = self.create_report(audio_issues, audio_metrics, system_checks)

        # Determine output path
        if output_dir is None:
            output_dir = "./logs/diagnostics"

        filename = f"diagnostic_report_{report.session_id}.json"
        output_path = os.path.join(output_dir, filename)

        report.save(output_path)

        return report


class ConsoleReporter:
    """Formats diagnostic reports for console output."""

    @staticmethod
    def format_report(report: DiagnosticReport) -> str:
        """Format report for console output.

        Args:
            report: DiagnosticReport to format

        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Diagnostic Report: {report.session_id}")
        lines.append(f"Generated: {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        # Summary
        summary = report.generate_summary()
        lines.append("\nSummary:")
        lines.append(f"  Overall Status: {summary['overall_status'].upper()}")
        lines.append(f"  Total Issues: {summary['total_issues']}")

        if summary['severity_breakdown']:
            lines.append("  Severity Breakdown:")
            for severity, count in summary['severity_breakdown'].items():
                if count > 0:
                    lines.append(f"    {severity.upper()}: {count}")

        # Issues by severity
        if report.get_critical_issues():
            lines.append("\n" + "-" * 60)
            lines.append("CRITICAL ISSUES:")
            for issue in report.get_critical_issues():
                lines.append(ConsoleReporter.format_issue(issue))

        if report.get_error_issues():
            lines.append("\n" + "-" * 60)
            lines.append("ERRORS:")
            for issue in report.get_error_issues():
                lines.append(ConsoleReporter.format_issue(issue))

        if report.get_warning_issues():
            lines.append("\n" + "-" * 60)
            lines.append("WARNINGS:")
            for issue in report.get_warning_issues():
                lines.append(ConsoleReporter.format_issue(issue))

        # Audio metrics
        if report.audio_metrics:
            lines.append("\n" + "-" * 60)
            lines.append("Audio Metrics:")
            metrics = report.audio_metrics
            if metrics.rms_level is not None:
                lines.append(f"  RMS Level: {metrics.rms_level:.4f}")
            if metrics.peak_level is not None:
                lines.append(f"  Peak Level: {metrics.peak_level:.4f}")
            if metrics.snr_db is not None:
                lines.append(f"  SNR: {metrics.snr_db:.1f} dB")
            if metrics.echo_detected:
                lines.append(f"  Echo: DETECTED ({metrics.echo_delay_ms:.1f}ms delay)")
            if metrics.crosstalk_detected:
                lines.append(f"  Crosstalk: DETECTED")

        # System checks
        if report.system_checks:
            lines.append("\n" + "-" * 60)
            lines.append("System Checks:")
            for check in report.system_checks:
                status_symbol = {
                    CheckStatus.PASSED: "✓",
                    CheckStatus.FAILED: "✗",
                    CheckStatus.WARNING: "⚠",
                    CheckStatus.SKIPPED: "⊘"
                }.get(check.status, "?")
                lines.append(f"  {status_symbol} {check.component.value}: {check.message}")

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def format_issue(issue: DiagnosticIssue) -> str:
        """Format a single issue for console output.

        Args:
            issue: DiagnosticIssue to format

        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"  [{issue.severity.value.upper()}] {issue.component}: {issue.message}")

        if issue.fix_suggestions:
            lines.append("    Suggested fixes:")
            for fix in sorted(issue.fix_suggestions, key=lambda f: f.priority):
                lines.append(f"      - {fix.action}")
                if fix.description:
                    lines.append(f"        {fix.description}")

        return "\n".join(lines)


def print_report_to_console(report: DiagnosticReport) -> None:
    """Print a diagnostic report to console.

    Args:
        report: DiagnosticReport to print
    """
    print(ConsoleReporter.format_report(report))


async def generate_diagnostic_report(
    audio_issues: Optional[List[AudioIssue]] = None,
    audio_metrics: Optional[AudioMetrics] = None,
    system_checks: Optional[List[CheckResult]] = None,
    session_id: Optional[str] = None,
    save_to: Optional[str] = None
) -> DiagnosticReport:
    """Generate a diagnostic report.

    Args:
        audio_issues: List of audio issues found
        audio_metrics: Audio metrics collected
        system_checks: List of system check results
        session_id: Optional session identifier
        save_to: Optional path to save report

    Returns:
        DiagnosticReport object
    """
    generator = ReportGenerator(session_id=session_id)

    if save_to:
        return await generator.generate_and_save_report(
            audio_issues, audio_metrics, system_checks, save_to
        )
    else:
        return generator.create_report(
            audio_issues, audio_metrics, system_checks
        )
