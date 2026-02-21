"""Diagnostics module for audio and system health checks.

This module provides comprehensive diagnostic capabilities for:
- Audio echo and crosstalk detection
- Audio quality analysis (SNR, latency, clipping)
- System health checks (microphone, speaker, network, performance)
- Diagnostic report generation with fix suggestions

Example usage:
    from diagnostics import (
        AudioDiagnostics,
        SystemDiagnostics,
        generate_diagnostic_report
    )

    # Audio diagnostics
    audio_diag = await create_audio_diagnostics()
    metrics, issues = await audio_diag.check_audio_loopback(
        played_audio,
        captured_audio
    )

    # System diagnostics
    system_diag = await create_system_diagnostics()
    check_results = await system_diag.run_all_checks()

    # Generate report
    report = await generate_diagnostic_report(
        audio_issues=issues,
        audio_metrics=metrics,
        system_checks=check_results
    )
    print_report_to_console(report)
"""

from .audio_check import (
    AudioIssue,
    AudioIssueType,
    AudioMetrics,
    AudioDiagnostics,
    AudioDiagnosticsError,
    EchoDetector,
    CrosstalkDetector,
    AudioQualityAnalyzer,
    create_audio_diagnostics,
)

from .system_check import (
    CheckStatus,
    CheckResult,
    SystemComponent,
    SystemInfo,
    DeviceInfo,
    AudioDeviceChecker,
    NetworkChecker,
    PerformanceChecker,
    PermissionChecker,
    SystemDiagnostics,
    SystemDiagnosticsError,
    create_system_diagnostics,
)

from .report import (
    SeverityLevel,
    FixSuggestion,
    DiagnosticIssue,
    DiagnosticReport,
    IssueClassifier,
    ReportGenerator,
    ConsoleReporter,
    generate_diagnostic_report,
    print_report_to_console,
)

__all__ = [
    # Audio diagnostics
    'AudioIssue',
    'AudioIssueType',
    'AudioMetrics',
    'AudioDiagnostics',
    'AudioDiagnosticsError',
    'EchoDetector',
    'CrosstalkDetector',
    'AudioQualityAnalyzer',
    'create_audio_diagnostics',
    # System diagnostics
    'CheckStatus',
    'CheckResult',
    'SystemComponent',
    'SystemInfo',
    'DeviceInfo',
    'AudioDeviceChecker',
    'NetworkChecker',
    'PerformanceChecker',
    'PermissionChecker',
    'SystemDiagnostics',
    'SystemDiagnosticsError',
    'create_system_diagnostics',
    # Reporting
    'SeverityLevel',
    'FixSuggestion',
    'DiagnosticIssue',
    'DiagnosticReport',
    'IssueClassifier',
    'ReportGenerator',
    'ConsoleReporter',
    'generate_diagnostic_report',
    'print_report_to_console',
]
