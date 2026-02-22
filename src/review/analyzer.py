"""Weakness analysis engine for review functionality."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from review.stats import SessionStatistics
from utils.logger import get_logger

logger = get_logger(__name__)


# Common English gaming phrases for pattern matching
COMMON_PHRASES = {
    "cover_me": "Cover me",
    "watch_out": "Watch out",
    "enemy_detected": "Enemy detected",
    "need_healing": "Need healing",
    "low_hp": "Low HP",
    "reload": "Reload",
    "push": "Push",
    "retreat": "Retreat",
    "defend": "Defend",
    "rotate": "Rotate",
    "high_ground": "High ground",
    "build": "Build",
    "edit": "Edit",
    "crank": "Crank 90s",
    "box_up": "Box up",
    "third_party": "Third party",
    "one_by_one": "One by one",
    "take_height": "Take height",
    "loot": "Loot",
    "shield": "Shield",
    "health": "Health",
    "storm": "Storm",
    "zone": "Zone",
}


@dataclass
class PhrasePattern:
    """A repeated phrase pattern."""

    phrase: str
    count: int
    contexts: list[str] = field(default_factory=list)
    first_used: int = 0  # timestamp_ms
    last_used: int = 0  # timestamp_ms


@dataclass
class WeaknessPoint:
    """An identified weakness area."""

    category: str
    severity: str  # "critical", "moderate", "minor"
    description: str
    suggestion: str
    evidence: list[str] = field(default_factory=list)
    affected_speeches: int = 0


@dataclass
class StrengthPoint:
    """An identified strength area."""

    category: str
    description: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of weakness analysis."""

    weaknesses: list[WeaknessPoint]
    strengths: list[StrengthPoint]
    phrase_patterns: list[PhrasePattern]
    vocabulary_analysis: dict[str, Any]
    improvement_suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "weaknesses": [
                {
                    "category": w.category,
                    "severity": w.severity,
                    "description": w.description,
                    "suggestion": w.suggestion,
                    "evidence": w.evidence,
                    "affected_speeches": w.affected_speeches,
                }
                for w in self.weaknesses
            ],
            "strengths": [
                {
                    "category": s.category,
                    "description": s.description,
                    "evidence": s.evidence,
                }
                for s in self.strengths
            ],
            "phrase_patterns": [
                {
                    "phrase": p.phrase,
                    "count": p.count,
                    "contexts": p.contexts,
                    "first_used": p.first_used,
                    "last_used": p.last_used,
                }
                for p in self.phrase_patterns
            ],
            "vocabulary_analysis": self.vocabulary_analysis,
            "improvement_suggestions": self.improvement_suggestions,
        }


class WeaknessAnalyzer:
    """Analyze session data to identify weaknesses and improvement areas."""

    def __init__(self):
        """Initialize weakness analyzer."""
        self.logger = logger

    async def analyze(self, stats: SessionStatistics) -> AnalysisResult:
        """
        Analyze session statistics to identify weaknesses.

        Args:
            stats: Collected session statistics

        Returns:
            Analysis result with weaknesses and suggestions
        """
        try:
            # Analyze phrase patterns
            phrase_patterns = await self._analyze_phrase_patterns(stats)

            # Analyze vocabulary usage
            vocab_analysis = await self._analyze_vocabulary(stats)

            # Identify weaknesses
            weaknesses = await self._identify_weaknesses(stats, vocab_analysis, phrase_patterns)

            # Identify strengths
            strengths = await self._identify_strengths(stats, vocab_analysis)

            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(weaknesses, vocab_analysis)

            return AnalysisResult(
                weaknesses=weaknesses,
                strengths=strengths,
                phrase_patterns=phrase_patterns,
                vocabulary_analysis=vocab_analysis,
                improvement_suggestions=suggestions,
            )

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}", e)
            raise

    async def _analyze_phrase_patterns(self, stats: SessionStatistics) -> list[PhrasePattern]:
        """
        Analyze repeated phrase patterns in speeches.

        Args:
            stats: Session statistics

        Returns:
            List of phrase patterns
        """
        phrase_counter = defaultdict(lambda: {"count": 0, "contexts": [], "first": None, "last": None})

        for speech in stats.speeches:
            response_text = speech.get("response", "").lower()
            timestamp_ms = speech.get("timestamp_ms", 0)
            trigger_name = speech.get("trigger_name", "unknown")

            # Check for common gaming phrases
            for phrase_key, phrase_text in COMMON_PHRASES.items():
                if phrase_text.lower() in response_text:
                    phrase_counter[phrase_key]["count"] += 1
                    phrase_counter[phrase_key]["contexts"].append(trigger_name)
                    if phrase_counter[phrase_key]["first"] is None:
                        phrase_counter[phrase_key]["first"] = timestamp_ms
                    phrase_counter[phrase_key]["last"] = timestamp_ms

        # Convert to PhrasePattern objects
        patterns = []
        for phrase_key, data in phrase_counter.items():
            if data["count"] >= 2:  # Only include repeated phrases
                patterns.append(PhrasePattern(
                    phrase=COMMON_PHRASES[phrase_key],
                    count=data["count"],
                    contexts=data["contexts"],
                    first_used=data["first"],
                    last_used=data["last"],
                ))

        # Sort by count (most frequent first)
        patterns.sort(key=lambda p: p.count, reverse=True)

        return patterns

    async def _analyze_vocabulary(self, stats: SessionStatistics) -> dict[str, Any]:
        """
        Analyze vocabulary usage patterns.

        Args:
            stats: Session statistics

        Returns:
            Vocabulary analysis dictionary
        """
        # Word frequency analysis
        word_freq = stats.vocabulary_frequency.copy()

        # Find overused words (words used more than 10% of total words)
        overused = []
        if stats.total_word_count > 0:
            for word, count in word_freq.items():
                ratio = count / stats.total_word_count
                if ratio > 0.10 and count > 2:  # Used >10% of the time and at least twice
                    overused.append({"word": word, "count": count, "ratio": round(ratio, 3)})

        # Find rare words (used once)
        rare_words = [word for word, count in word_freq.items() if count == 1]

        # Analyze word complexity (simple heuristic: longer words = more complex)
        word_lengths = [len(word) for word in word_freq.keys()]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0

        # Categorize vocabulary by game context
        combat_words = {"enemy", "shoot", "kill", "fight", "damage", "attack", "defense", "cover"}
        building_words = {"build", "ramp", "floor", "wall", "roof", "edit", "box"}
        movement_words = {"run", "walk", "jump", "rotate", "move", "push", "retreat"}
        resource_words = {"shield", "health", "ammo", "material", "wood", "brick", "metal", "loot"}

        context_usage = {
            "combat": sum(word_freq.get(w, 0) for w in combat_words),
            "building": sum(word_freq.get(w, 0) for w in building_words),
            "movement": sum(word_freq.get(w, 0) for w in movement_words),
            "resource": sum(word_freq.get(w, 0) for w in resource_words),
        }

        return {
            "unique_words": len(stats.unique_vocabulary),
            "total_words": stats.total_word_count,
            "avg_word_length": round(avg_word_length, 2),
            "overused_words": sorted(overused, key=lambda x: x["count"], reverse=True)[:5],
            "rare_words_count": len(rare_words),
            "context_distribution": context_usage,
            "diversity_score": round(len(stats.unique_vocabulary) / max(stats.total_word_count, 1), 3),
        }

    async def _identify_weaknesses(
        self,
        stats: SessionStatistics,
        vocab_analysis: dict[str, Any],
        phrase_patterns: list[PhrasePattern]
    ) -> list[WeaknessPoint]:
        """
        Identify specific weakness areas.

        Args:
            stats: Session statistics
            vocab_analysis: Vocabulary analysis results
            phrase_patterns: Phrase pattern analysis results

        Returns:
            List of identified weaknesses
        """
        weaknesses = []

        # Check for low vocabulary diversity
        if vocab_analysis["diversity_score"] < 0.4:
            weaknesses.append(WeaknessPoint(
                category="vocabulary",
                severity="moderate" if vocab_analysis["diversity_score"] > 0.2 else "critical",
                description="Low vocabulary diversity detected. Repeating the same words frequently.",
                suggestion="Try to use more varied vocabulary. Practice different ways to express similar concepts.",
                evidence=[f"Diversity score: {vocab_analysis['diversity_score']:.2f} (target: >0.5)"],
                affected_speeches=stats.total_speeches,
            ))

        # Check for overused words
        if vocab_analysis["overused_words"]:
            overused_list = [w["word"] for w in vocab_analysis["overused_words"][:3]]
            weaknesses.append(WeaknessPoint(
                category="vocabulary",
                severity="minor",
                description=f"Over-reliance on certain words: {', '.join(overused_list)}",
                suggestion="Practice synonyms and alternative expressions for these words.",
                evidence=[f"'{w['word']}' used {w['count']} times" for w in vocab_analysis["overused_words"][:3]],
                affected_speeches=sum(w["count"] for w in vocab_analysis["overused_words"]),
            ))

        # Check for low speech frequency (not coaching enough)
        if stats.total_speeches < 5:
            weaknesses.append(WeaknessPoint(
                category="engagement",
                severity="moderate",
                description=f"Low coaching frequency. Only {stats.total_speeches} speech events recorded.",
                suggestion="Provide more frequent coaching feedback to keep the player engaged and informed.",
                evidence=["Total speeches: " + str(stats.total_speeches)],
                affected_speeches=stats.total_speeches,
            ))

        # Check for lack of combat awareness
        if stats.total_speeches > 5 and stats.combat_speeches == 0:
            weaknesses.append(WeaknessPoint(
                category="situational_awareness",
                severity="moderate",
                description="No coaching during combat situations detected.",
                suggestion="Pay attention to combat situations and provide tactical guidance.",
                evidence=["Combat speeches: 0", "Non-combat speeches: " + str(stats.non_combat_speeches)],
                affected_speeches=0,
            ))

        # Check for slow response times
        if stats.avg_response_time_ms > 5000 and stats.response_times:
            weaknesses.append(WeaknessPoint(
                category="response_speed",
                severity="moderate",
                description=f"Slow response time: {stats.avg_response_time_ms / 1000:.1f}s average",
                suggestion="Aim to respond within 2-3 seconds of game events for more effective coaching.",
                evidence=[f"Average response time: {stats.avg_response_time_ms / 1000:.1f}s"],
                affected_speeches=len(stats.response_times),
            ))

        # Check for repetitive phrase usage
        if phrase_patterns and phrase_patterns[0].count > 5:
            top_pattern = phrase_patterns[0]
            weaknesses.append(WeaknessPoint(
                category="variety",
                severity="minor",
                description=f"Heavy repetition of phrase: '{top_pattern.phrase}'",
                suggestion="Vary your coaching phrases to keep communication fresh and engaging.",
                evidence=[f"Used {top_pattern.count} times"],
                affected_speeches=top_pattern.count,
            ))

        # Check for context imbalance
        context_dist = vocab_analysis.get("context_distribution", {})
        if context_dist:
            max_context = max(context_dist.values())
            min_context = min(context_dist.values())
            if max_context > 0 and min_context == 0:
                missing_contexts = [k for k, v in context_dist.items() if v == 0]
                weaknesses.append(WeaknessPoint(
                    category="context_coverage",
                    severity="minor",
                    description=f"Missing coaching for: {', '.join(missing_contexts)}",
                    suggestion="Ensure you provide coaching across all game contexts.",
                    evidence=[f"No {ctx}-related vocabulary" for ctx in missing_contexts],
                    affected_speeches=0,
                ))

        # Sort by severity (critical > moderate > minor)
        severity_order = {"critical": 0, "moderate": 1, "minor": 2}
        weaknesses.sort(key=lambda w: severity_order[w.severity])

        return weaknesses

    async def _identify_strengths(
        self,
        stats: SessionStatistics,
        vocab_analysis: dict[str, Any]
    ) -> list[StrengthPoint]:
        """
        Identify strength areas.

        Args:
            stats: Session statistics
            vocab_analysis: Vocabulary analysis results

        Returns:
            List of identified strengths
        """
        strengths = []

        # Check for good vocabulary diversity
        if vocab_analysis["diversity_score"] > 0.6:
            strengths.append(StrengthPoint(
                category="vocabulary",
                description="Excellent vocabulary diversity with varied word usage.",
                evidence=[f"Diversity score: {vocab_analysis['diversity_score']:.2f}"],
            ))

        # Check for good engagement
        if stats.total_speeches >= 15:
            strengths.append(StrengthPoint(
                category="engagement",
                description=f"Strong coaching engagement with {stats.total_speeches} speech events.",
                evidence=[f"Total speeches: {stats.total_speeches}"],
            ))

        # Check for good response speed
        if stats.avg_response_time_ms > 0 and stats.avg_response_time_ms < 3000:
            strengths.append(StrengthPoint(
                category="response_speed",
                description=f"Excellent response time: {stats.avg_response_time_ms / 1000:.1f}s average",
                evidence=[f"Average response time: {stats.avg_response_time_ms / 1000:.1f}s"],
            ))

        # Check for balanced combat/non-combat coverage
        if stats.total_speeches > 5:
            combat_ratio = stats.combat_speeches / stats.total_speeches
            if 0.3 <= combat_ratio <= 0.7:
                strengths.append(StrengthPoint(
                    category="balance",
                    description="Good balance between combat and non-combat coaching.",
                    evidence=[
                        f"Combat: {stats.combat_speeches}",
                        f"Non-combat: {stats.non_combat_speeches}",
                    ],
                ))

        # Check for good context coverage
        context_dist = vocab_analysis.get("context_distribution", {})
        if context_dist:
            active_contexts = [k for k, v in context_dist.items() if v > 0]
            if len(active_contexts) >= 3:
                strengths.append(StrengthPoint(
                    category="context_coverage",
                    description=f"Good coverage across {len(active_contexts)} game contexts.",
                    evidence=[f"Active contexts: {', '.join(active_contexts)}"],
                ))

        return strengths

    async def _generate_suggestions(
        self,
        weaknesses: list[WeaknessPoint],
        vocab_analysis: dict[str, Any]
    ) -> list[str]:
        """
        Generate actionable improvement suggestions.

        Args:
            weaknesses: Identified weaknesses
            vocab_analysis: Vocabulary analysis results

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Add suggestions from weaknesses
        for weakness in weaknesses:
            if weakness.suggestion not in suggestions:
                suggestions.append(weakness.suggestion)

        # Add general suggestions based on analysis
        if vocab_analysis["diversity_score"] < 0.5:
            suggestions.append(
                "Practice using 5-10 new vocabulary words each session to gradually expand your vocabulary."
            )

        # Add context-specific suggestions
        context_dist = vocab_analysis.get("context_distribution", {})
        if context_dist.get("building", 0) < context_dist.get("combat", 0):
            suggestions.append(
                "Consider incorporating more building-related coaching phrases like 'build cover', 'ramp up', 'edit through'."
            )

        # Add variety suggestion
        suggestions.append(
            "Prepare a set of alternative phrases for common situations to avoid repetition."
        )

        # Limit to top 5 most relevant suggestions
        return suggestions[:5]


async def analyze_session(session_dir: str) -> AnalysisResult:
    """
    Convenience function to analyze a session.

    Args:
        session_dir: Path to session directory

    Returns:
        Analysis result
    """
    from review.stats import SessionStatsCollector

    # Collect statistics
    collector = SessionStatsCollector(session_dir)
    stats = await collector.collect()

    # Analyze
    analyzer = WeaknessAnalyzer()
    return await analyzer.analyze(stats)
