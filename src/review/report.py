"""Review report generation for session analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from review.analyzer import AnalysisResult
from review.scorer import OverallScore
from review.stats import SessionStatistics
from utils.logger import get_logger

logger = get_logger(__name__)


class ReviewReportGenerator:
    """Generate review reports from session analysis data."""

    def __init__(self):
        """Initialize report generator."""
        self.logger = logger

    async def generate(
        self,
        stats: SessionStatistics,
        score: OverallScore,
        analysis: AnalysisResult,
        output_format: str = "both"
    ) -> Dict[str, str]:
        """
        Generate review report in specified format(s).

        Args:
            stats: Session statistics
            score: Calculated score
            analysis: Analysis results
            output_format: "text", "json", or "both"

        Returns:
            Dictionary with format as key and content/file_path as value
        """
        results = {}

        try:
            if output_format in ("text", "both"):
                text_report = self._generate_text_report(stats, score, analysis)
                results["text"] = text_report

            if output_format in ("json", "both"):
                json_report = self._generate_json_report(stats, score, analysis)
                results["json"] = json_report

            return results

        except Exception as e:
            self.logger.error(f"Error generating report: {e}", e)
            raise

    def _generate_text_report(
        self,
        stats: SessionStatistics,
        score: OverallScore,
        analysis: AnalysisResult
    ) -> str:
        """
        Generate human-readable text report.

        Args:
            stats: Session statistics
            score: Calculated score
            analysis: Analysis results

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SESSION REVIEW REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Session info
        lines.append("SESSION INFORMATION")
        lines.append("-" * 70)
        lines.append(f"Session Directory: {stats.session_dir}")
        lines.append(f"Start Time: {stats.start_time}")
        if stats.end_time:
            lines.append(f"End Time: {stats.end_time}")
        lines.append("")

        # Overall Score
        lines.append("OVERALL SCORE")
        lines.append("-" * 70)
        lines.append(f"Grade: {score.grade}")
        lines.append(f"Total Score: {score.total_score:.1f}/100")
        lines.append(f"Summary: {score.summary}")
        lines.append("")

        # Category Scores
        lines.append("CATEGORY SCORES")
        lines.append("-" * 70)
        for category in score.categories:
            lines.append(f"  {category.name.replace('_', ' ').title()}: "
                        f"{category.score:.1f}/100 (weight: {category.weight*100:.0f}%)")
        lines.append("")

        # Category Details
        lines.append("CATEGORY DETAILS")
        lines.append("-" * 70)
        for category in score.categories:
            lines.append(f"\n{category.name.replace('_', ' ').title()}:")
            for key, value in category.details.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"  - {key.replace('_', ' ').title()}: {value}")
                elif isinstance(value, list) and value:
                    lines.append(f"  - {key.replace('_', ' ').title()}:")
                    for item in value[:5]:  # Limit to 5 items
                        if isinstance(item, dict):
                            lines.append(f"    * {item}")
                        else:
                            lines.append(f"    * {item}")
        lines.append("")

        # Statistics Summary
        lines.append("SESSION STATISTICS")
        lines.append("-" * 70)
        lines.append(f"Total Speeches: {stats.total_speeches}")
        lines.append(f"Combat Speeches: {stats.combat_speeches}")
        lines.append(f"Non-Combat Speeches: {stats.non_combat_speeches}")
        lines.append(f"Total Words: {stats.total_word_count}")
        lines.append(f"Unique Vocabulary: {len(stats.unique_vocabulary)}")
        if stats.avg_response_time_ms > 0:
            lines.append(f"Avg Response Time: {stats.avg_response_time_ms / 1000:.1f}s")
        if stats.avg_voice_duration_ms > 0:
            lines.append(f"Avg Voice Duration: {stats.avg_voice_duration_ms / 1000:.1f}s")
        lines.append("")

        # Weaknesses
        if analysis.weaknesses:
            lines.append("AREAS FOR IMPROVEMENT")
            lines.append("-" * 70)
            for i, weakness in enumerate(analysis.weaknesses, 1):
                severity_symbol = {
                    "critical": "!!!",
                    "moderate": "!!",
                    "minor": "!",
                }.get(weakness.severity, "?")
                lines.append(f"{i}. [{severity_symbol}] {weakness.category.upper()}")
                lines.append(f"   Description: {weakness.description}")
                lines.append(f"   Suggestion: {weakness.suggestion}")
                if weakness.evidence:
                    lines.append(f"   Evidence:")
                    for evidence in weakness.evidence[:3]:
                        lines.append(f"     - {evidence}")
                lines.append("")

        # Strengths
        if analysis.strengths:
            lines.append("STRENGTHS")
            lines.append("-" * 70)
            for i, strength in enumerate(analysis.strengths, 1):
                lines.append(f"{i}. {strength.category.replace('_', ' ').title()}")
                lines.append(f"   {strength.description}")
                if strength.evidence:
                    for evidence in strength.evidence[:2]:
                        lines.append(f"   - {evidence}")
            lines.append("")

        # Phrase Patterns
        if analysis.phrase_patterns:
            lines.append("REPEATED PHRASES")
            lines.append("-" * 70)
            for pattern in analysis.phrase_patterns[:5]:
                lines.append(f"  '{pattern.phrase}' - used {pattern.count} times")
            lines.append("")

        # Vocabulary Analysis
        vocab = analysis.vocabulary_analysis
        lines.append("VOCABULARY ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"Unique Words: {vocab['unique_words']}")
        lines.append(f"Total Words: {vocab['total_words']}")
        lines.append(f"Diversity Score: {vocab['diversity_score']:.2f}")
        lines.append(f"Avg Word Length: {vocab['avg_word_length']:.1f} characters")
        if vocab.get("overused_words"):
            lines.append("Overused Words:")
            for word_data in vocab["overused_words"][:3]:
                lines.append(f"  - '{word_data['word']}' used {word_data['count']} times "
                           f"({word_data['ratio']*100:.1f}% of all words)")
        lines.append("")

        # Improvement Suggestions
        if analysis.improvement_suggestions:
            lines.append("IMPROVEMENT SUGGESTIONS")
            lines.append("-" * 70)
            for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
                lines.append(f"{i}. {suggestion}")
        lines.append("")

        lines.append("=" * 70)
        lines.append(f"Report Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_json_report(
        self,
        stats: SessionStatistics,
        score: OverallScore,
        analysis: AnalysisResult
    ) -> str:
        """
        Generate JSON format report.

        Args:
            stats: Session statistics
            score: Calculated score
            analysis: Analysis results

        Returns:
            JSON string report
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_dir": stats.session_dir,
                "start_time": stats.start_time,
                "end_time": stats.end_time,
            },
            "score": score.to_dict(),
            "statistics": stats.to_dict(),
            "analysis": analysis.to_dict(),
        }

        return json.dumps(report, indent=2, ensure_ascii=False)

    async def save_report(
        self,
        content: str,
        output_path: str,
        format_type: str = "text"
    ) -> str:
        """
        Save report to file.

        Args:
            content: Report content
            output_path: Output file path
            format_type: Format type for logging

        Returns:
            Absolute path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w') as f:
                f.write(content)

            self.logger.info(f"{format_type.upper()} report saved to: {output_path}")
            return str(output_path.absolute())

        except IOError as e:
            self.logger.error(f"Error saving {format_type} report: {e}", e)
            raise


async def generate_review_report(
    session_dir: str,
    output_dir: Optional[str] = None,
    output_format: str = "both"
) -> Dict[str, str]:
    """
    Convenience function to generate a complete review report.

    Args:
        session_dir: Path to session directory
        output_dir: Optional output directory (defaults to session_dir)
        output_format: "text", "json", or "both"

    Returns:
        Dictionary with format as key and file path as value
    """
    from review.analyzer import WeaknessAnalyzer
    from review.scorer import ScoreCalculator
    from review.stats import SessionStatsCollector

    # Collect statistics
    collector = SessionStatsCollector(session_dir)
    stats = await collector.collect()

    # Calculate score
    calculator = ScoreCalculator()
    score = await calculator.calculate(stats)

    # Analyze weaknesses
    analyzer = WeaknessAnalyzer()
    analysis = await analyzer.analyze(stats)

    # Generate report
    generator = ReviewReportGenerator()
    results = await generator.generate(stats, score, analysis, output_format)

    # Save reports
    if output_dir is None:
        output_dir = session_dir

    output_dir = Path(output_dir)
    saved_paths = {}

    if "text" in results:
        text_path = output_dir / "review_report.txt"
        saved_paths["text"] = await generator.save_report(
            results["text"], str(text_path), "text"
        )

    if "json" in results:
        json_path = output_dir / "review_report.json"
        saved_paths["json"] = await generator.save_report(
            results["json"], str(json_path), "json"
        )

    return saved_paths


async def generate_session_summary(session_dir: str) -> str:
    """
    Generate a brief text summary of a session.

    Args:
        session_dir: Path to session directory

    Returns:
        Brief summary text
    """
    from review.stats import SessionStatsCollector

    # Load statistics
    try:
        stats_path = Path(session_dir) / "statistics.json"
        if stats_path.exists():
            from review.stats import load_statistics
            stats = await load_statistics(str(stats_path))
        else:
            collector = SessionStatsCollector(session_dir)
            stats = await collector.collect()
    except Exception as e:
        logger.error(f"Error loading statistics: {e}")
        return f"Unable to load session statistics: {e}"

    # Generate summary
    summary_lines = [
        f"Session Summary ({Path(session_dir).name})",
        f"  Total Speeches: {stats.total_speeches}",
        f"  Combat: {stats.combat_speeches} | Non-Combat: {stats.non_combat_speeches}",
        f"  Total Words: {stats.total_word_count} | Unique: {len(stats.unique_vocabulary)}",
    ]

    if stats.avg_response_time_ms > 0:
        summary_lines.append(f"  Avg Response Time: {stats.avg_response_time_ms / 1000:.1f}s")

    if stats.avg_voice_duration_ms > 0:
        summary_lines.append(f"  Avg Voice Duration: {stats.avg_voice_duration_ms / 1000:.1f}s")

    return "\n".join(summary_lines)
