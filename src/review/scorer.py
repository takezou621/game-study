"""Score calculation logic for review functionality."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from review.stats import SessionStatistics
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CategoryScore:
    """Score for a specific category."""

    name: str
    score: float  # 0-100
    weight: float  # Weight in overall score
    details: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.name}: {self.score:.1f}/100"


@dataclass
class OverallScore:
    """Overall score with category breakdown."""

    total_score: float  # 0-100
    categories: List[CategoryScore]
    grade: str  # A, B, C, D, F
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_score": round(self.total_score, 1),
            "grade": self.grade,
            "summary": self.summary,
            "categories": [
                {
                    "name": cat.name,
                    "score": round(cat.score, 1),
                    "weight": cat.weight,
                    "details": cat.details,
                }
                for cat in self.categories
            ],
        }


class ScoreCalculator:
    """Calculate scores from session statistics."""

    # Score thresholds for grading
    GRADE_THRESHOLDS = {
        "A": 90,
        "B": 75,
        "C": 60,
        "D": 40,
        "F": 0,
    }

    # Category weights for overall score
    CATEGORY_WEIGHTS = {
        "pronunciation": 0.20,
        "vocabulary": 0.25,
        "response_speed": 0.20,
        "strategic_thinking": 0.35,
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize score calculator.

        Args:
            custom_weights: Optional custom category weights
        """
        if custom_weights:
            self.CATEGORY_WEIGHTS = custom_weights
        self.logger = logger

    async def calculate(self, stats: SessionStatistics) -> OverallScore:
        """
        Calculate overall score from session statistics.

        Args:
            stats: Collected session statistics

        Returns:
            Overall score with category breakdown
        """
        try:
            # Calculate individual category scores
            categories = [
                await self._calculate_pronunciation_score(stats),
                await self._calculate_vocabulary_score(stats),
                await self._calculate_response_speed_score(stats),
                await self._calculate_strategic_thinking_score(stats),
            ]

            # Calculate weighted total score
            total_score = sum(
                cat.score * cat.weight
                for cat in categories
            )

            # Determine grade
            grade = self._determine_grade(total_score)

            # Generate summary
            summary = self._generate_summary(total_score, categories)

            return OverallScore(
                total_score=total_score,
                categories=categories,
                grade=grade,
                summary=summary,
            )

        except Exception as e:
            self.logger.error(f"Error calculating score: {e}", e)
            raise

    async def _calculate_pronunciation_score(self, stats: SessionStatistics) -> CategoryScore:
        """
        Calculate pronunciation score.

        Note: This is a placeholder for future integration with
        actual pronunciation analysis. Currently uses proxy metrics.

        Args:
            stats: Session statistics

        Returns:
            Pronunciation category score
        """
        # Base score from voice output consistency
        base_score = 70.0

        # Adjust based on voice duration consistency
        if stats.avg_voice_duration_ms > 0:
            speeches_with_voice = sum(
                1 for s in stats.speeches
                if s.get("voice_duration_ms") is not None
            )
            voice_coverage = speeches_with_voice / max(stats.total_speeches, 1)
            base_score += voice_coverage * 20

        # Bonus for consistent voice output
        if stats.avg_voice_duration_ms > 0:
            duration_variance_penalty = 0
            if len(stats.speeches) > 1:
                durations = [
                    s.get("voice_duration_ms", 0)
                    for s in stats.speeches
                    if s.get("voice_duration_ms") is not None
                ]
                if durations:
                    avg = sum(durations) / len(durations)
                    variance = sum((d - avg) ** 2 for d in durations) / len(durations)
                    # Penalize high variance
                    duration_variance_penalty = min(variance / 1000000, 10)

            base_score -= duration_variance_penalty

        # Clamp score to 0-100
        score = max(0, min(100, base_score))

        return CategoryScore(
            name="pronunciation",
            score=score,
            weight=self.CATEGORY_WEIGHTS["pronunciation"],
            details={
                "voice_coverage_percent": round(
                    (sum(
                        1 for s in stats.speeches
                        if s.get("voice_duration_ms") is not None
                    ) / max(stats.total_speeches, 1)) * 100, 1
                ),
                "avg_voice_duration_ms": round(stats.avg_voice_duration_ms, 1),
                "note": "Pronunciation analysis requires audio input integration",
            },
        )

    async def _calculate_vocabulary_score(self, stats: SessionStatistics) -> CategoryScore:
        """
        Calculate vocabulary score.

        Args:
            stats: Session statistics

        Returns:
            Vocabulary category score
        """
        # Base score
        base_score = 50.0

        # Unique vocabulary size score (0-30 points)
        vocab_size = len(stats.unique_vocabulary)
        vocab_score = min(vocab_size * 2, 30)  # Cap at 30 points
        base_score += vocab_score

        # Vocabulary diversity (unique/total ratio) (0-20 points)
        if stats.total_word_count > 0:
            diversity_ratio = vocab_size / stats.total_word_count
            diversity_score = min(diversity_ratio * 100, 20)
            base_score += diversity_score

        # Average words per speech (0-10 bonus points)
        if stats.total_speeches > 0:
            avg_words_per_speech = stats.total_word_count / stats.total_speeches
            if avg_words_per_speech >= 5:
                base_score += 10

        # Clamp score to 0-100
        score = max(0, min(100, base_score))

        return CategoryScore(
            name="vocabulary",
            score=score,
            weight=self.CATEGORY_WEIGHTS["vocabulary"],
            details={
                "unique_words": vocab_size,
                "total_words": stats.total_word_count,
                "diversity_ratio": round(
                    vocab_size / max(stats.total_word_count, 1), 2
                ),
                "avg_words_per_speech": round(
                    stats.total_word_count / max(stats.total_speeches, 1), 1
                ),
                "top_words": self._get_top_words(stats.vocabulary_frequency, 5),
            },
        )

    async def _calculate_response_speed_score(self, stats: SessionStatistics) -> CategoryScore:
        """
        Calculate response speed score.

        Args:
            stats: Session statistics

        Returns:
            Response speed category score
        """
        # Base score
        base_score = 50.0

        if stats.response_times:
            avg_time = stats.avg_response_time_ms

            # Score based on average response time (lower is better)
            # Optimal: < 2 seconds = full points
            # Acceptable: < 5 seconds = partial points
            # Slow: > 5 seconds = minimal points
            if avg_time < 2000:
                time_score = 50
            elif avg_time < 5000:
                time_score = 50 - ((avg_time - 2000) / 3000) * 30
            else:
                time_score = 20 - min((avg_time - 5000) / 10000, 20)

            base_score += time_score

            # Consistency bonus (low variance is good)
            if len(stats.response_times) > 1:
                avg = sum(stats.response_times) / len(stats.response_times)
                variance = sum((t - avg) ** 2 for t in stats.response_times) / len(stats.response_times)
                std_dev = variance ** 0.5

                if std_dev < avg * 0.3:  # Low variance
                    base_score += 10
                elif std_dev < avg * 0.5:  # Medium variance
                    base_score += 5
        else:
            # No response time data, give neutral score
            base_score = 60.0

        # Clamp score to 0-100
        score = max(0, min(100, base_score))

        details = {
            "avg_response_time_ms": round(stats.avg_response_time_ms, 1),
            "min_response_time_ms": stats.min_response_time_ms,
            "max_response_time_ms": stats.max_response_time_ms,
        }

        # Add interpretation
        if stats.response_times:
            if stats.avg_response_time_ms < 2000:
                details["interpretation"] = "Excellent - very responsive"
            elif stats.avg_response_time_ms < 5000:
                details["interpretation"] = "Good - reasonable response time"
            else:
                details["interpretation"] = "Slow - consider improving response speed"
        else:
            details["interpretation"] = "Insufficient data for analysis"

        return CategoryScore(
            name="response_speed",
            score=score,
            weight=self.CATEGORY_WEIGHTS["response_speed"],
            details=details,
        )

    async def _calculate_strategic_thinking_score(self, stats: SessionStatistics) -> CategoryScore:
        """
        Calculate strategic thinking score.

        Args:
            stats: Session statistics

        Returns:
            Strategic thinking category score
        """
        # Base score
        base_score = 50.0

        # Combat vs non-combat balance (0-15 points)
        if stats.total_speeches > 0:
            combat_ratio = stats.combat_speeches / stats.total_speeches
            # Ideal ratio is around 0.3-0.7 (balanced)
            if 0.3 <= combat_ratio <= 0.7:
                balance_score = 15
            else:
                # Penalty for being too skewed
                balance_score = 15 - abs(combat_ratio - 0.5) * 30
            base_score += max(0, balance_score)

        # Priority diversity (0-20 points)
        total_priority = stats.high_priority_count + stats.medium_priority_count + stats.low_priority_count
        if total_priority > 0:
            # Having all priority levels indicates good situation awareness
            has_all_levels = (
                stats.high_priority_count > 0 and
                stats.medium_priority_count > 0 and
                stats.low_priority_count > 0
            )
            if has_all_levels:
                base_score += 20
            elif stats.high_priority_count > 0 and stats.medium_priority_count > 0:
                base_score += 15
            elif stats.high_priority_count > 0:
                base_score += 10

        # Trigger variety (0-15 points)
        unique_triggers = len(stats.trigger_counts)
        variety_score = min(unique_triggers * 3, 15)
        base_score += variety_score

        # Frequency of appropriate responses (0-20 bonus points)
        if stats.total_speeches > 10:
            # Good frequency of coaching
            frequency_score = 20
        elif stats.total_speeches > 5:
            frequency_score = 15
        elif stats.total_speeches > 0:
            frequency_score = 10
        else:
            frequency_score = 0
        base_score += frequency_score

        # Clamp score to 0-100
        score = max(0, min(100, base_score))

        return CategoryScore(
            name="strategic_thinking",
            score=score,
            weight=self.CATEGORY_WEIGHTS["strategic_thinking"],
            details={
                "total_speeches": stats.total_speeches,
                "combat_speeches": stats.combat_speeches,
                "non_combat_speeches": stats.non_combat_speeches,
                "high_priority_count": stats.high_priority_count,
                "medium_priority_count": stats.medium_priority_count,
                "low_priority_count": stats.low_priority_count,
                "unique_triggers": unique_triggers,
                "trigger_distribution": {
                    name: count
                    for name, count in sorted(
                        stats.trigger_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                },
            },
        )

    def _determine_grade(self, score: float) -> str:
        """
        Determine letter grade from score.

        Args:
            score: Numeric score (0-100)

        Returns:
            Letter grade (A, B, C, D, F)
        """
        if score >= self.GRADE_THRESHOLDS["A"]:
            return "A"
        elif score >= self.GRADE_THRESHOLDS["B"]:
            return "B"
        elif score >= self.GRADE_THRESHOLDS["C"]:
            return "C"
        elif score >= self.GRADE_THRESHOLDS["D"]:
            return "D"
        else:
            return "F"

    def _generate_summary(self, total_score: float, categories: List[CategoryScore]) -> str:
        """
        Generate summary text for the score.

        Args:
            total_score: Overall score
            categories: Category scores

        Returns:
            Summary text
        """
        # Find strongest and weakest categories
        sorted_cats = sorted(categories, key=lambda c: c.score, reverse=True)
        strongest = sorted_cats[0]
        weakest = sorted_cats[-1]

        grade = self._determine_grade(total_score)

        if grade == "A":
            overall = "Excellent performance!"
        elif grade == "B":
            overall = "Good performance with room for improvement."
        elif grade == "C":
            overall = "Fair performance. Focus on weaker areas."
        elif grade == "D":
            overall = "Needs significant improvement."
        else:
            overall = "Poor performance. Significant practice needed."

        return (
            f"{overall} "
            f"Strongest area: {strongest.name} ({strongest.score:.1f}/100). "
            f"Focus on improving: {weakest.name} ({weakest.score:.1f}/100)."
        )

    def _get_top_words(self, frequency: Dict[str, int], limit: int) -> List[Dict[str, Any]]:
        """
        Get top most frequent words.

        Args:
            frequency: Word frequency dictionary
            limit: Maximum number of words to return

        Returns:
            List of top words with counts
        """
        return [
            {"word": word, "count": count}
            for word, count in sorted(
                frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
        ]
