"""Review functionality for session analysis and feedback.

This module provides comprehensive review capabilities including:
- Session statistics collection and storage
- Score calculation across multiple categories
- Weakness analysis and improvement suggestions
- Report generation in multiple formats
"""

from review.analyzer import (
    AnalysisResult,
    PhrasePattern,
    StrengthPoint,
    WeaknessAnalyzer,
    WeaknessPoint,
    analyze_session,
)
from review.report import (
    ReviewReportGenerator,
    generate_review_report,
    generate_session_summary,
)
from review.scorer import CategoryScore, OverallScore, ScoreCalculator
from review.stats import (
    SessionStatistics,
    SessionStatsCollector,
    SpeechMetrics,
    load_statistics,
)

__all__ = [
    # Statistics
    "SessionStatistics",
    "SessionStatsCollector",
    "SpeechMetrics",
    "load_statistics",
    # Scoring
    "CategoryScore",
    "OverallScore",
    "ScoreCalculator",
    # Analysis
    "WeaknessAnalyzer",
    "WeaknessPoint",
    "StrengthPoint",
    "PhrasePattern",
    "AnalysisResult",
    "analyze_session",
    # Reports
    "ReviewReportGenerator",
    "generate_review_report",
    "generate_session_summary",
]
