"""Tests for review module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

try:
    from review.analyzer import (
        AnalysisResult,
        COMMON_PHRASES,
        PhrasePattern,
        StrengthPoint,
        WeaknessAnalyzer,
        WeaknessPoint,
        analyze_session,
    )
except ImportError:
    pytest.skip("review.analyzer module not available", allow_module_level=True)

try:
    from review.stats import (
        SessionStatistics,
        SessionStatsCollector,
        SpeechMetrics,
        load_statistics,
    )
except ImportError:
    pytest.skip("review.stats module not available", allow_module_level=True)

try:
    from review.scorer import (
        CategoryScore,
        OverallScore,
        ScoreCalculator,
    )
except ImportError:
    pytest.skip("review.scorer module not available", allow_module_level=True)

try:
    from review.report import (
        ReviewReportGenerator,
        generate_review_report,
        generate_session_summary,
    )
except ImportError:
    pytest.skip("review.report module not available", allow_module_level=True)


# ============================================================================
# PhrasePattern Tests
# ============================================================================

class TestPhrasePattern:
    """Test PhrasePattern dataclass."""

    def test_init(self):
        """Test initialization."""
        pattern = PhrasePattern(
            phrase="Test phrase",
            count=5,
            contexts=["combat", "non_combat"],
            first_used=1000,
            last_used=5000
        )

        assert pattern.phrase == "Test phrase"
        assert pattern.count == 5
        assert len(pattern.contexts) == 2


# ============================================================================
# WeaknessPoint Tests
# ============================================================================

class TestWeaknessPoint:
    """Test WeaknessPoint dataclass."""

    def test_init(self):
        """Test initialization."""
        weakness = WeaknessPoint(
            category="vocabulary",
            severity="moderate",
            description="Low diversity",
            suggestion="Use more words",
            evidence=["test evidence"],
            affected_speeches=5
        )

        assert weakness.category == "vocabulary"
        assert weakness.severity == "moderate"


# ============================================================================
# StrengthPoint Tests
# ============================================================================

class TestStrengthPoint:
    """Test StrengthPoint dataclass."""

    def test_init(self):
        """Test initialization."""
        strength = StrengthPoint(
            category="vocabulary",
            description="Good diversity",
            evidence=["evidence 1", "evidence 2"]
        )

        assert strength.category == "vocabulary"
        assert len(strength.evidence) == 2


# ============================================================================
# AnalysisResult Tests
# ============================================================================

class TestAnalysisResult:
    """Test AnalysisResult dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        result = AnalysisResult(
            weaknesses=[],
            strengths=[],
            phrase_patterns=[],
            vocabulary_analysis={"unique_words": 100},
            improvement_suggestions=["Practice more"]
        )

        result_dict = result.to_dict()

        assert result_dict["vocabulary_analysis"]["unique_words"] == 100
        assert len(result_dict["improvement_suggestions"]) == 1


# ============================================================================
# WeaknessAnalyzer Tests
# ============================================================================

class TestWeaknessAnalyzerInit:
    """Test WeaknessAnalyzer initialization."""

    def test_init(self):
        """Test initialization."""
        analyzer = WeaknessAnalyzer()
        assert analyzer.logger is not None


class TestWeaknessAnalyzerAnalyze:
    """Test WeaknessAnalyzer analyze methods."""

    @pytest.mark.asyncio
    async def test_analyze_empty_session(self):
        """Test analyzing empty session."""
        analyzer = WeaknessAnalyzer()

        stats = SessionStatistics(session_dir="/test")
        result = await analyzer.analyze(stats)

        assert isinstance(result, AnalysisResult)
        assert isinstance(result.weaknesses, list)
        assert isinstance(result.strengths, list)

    @pytest.mark.asyncio
    async def test_analyze_with_speeches(self):
        """Test analyzing session with speeches."""
        analyzer = WeaknessAnalyzer()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 10
        stats.speeches = [
            {
                "timestamp_ms": i * 1000,
                "response": f"Test response {i}",
                "trigger_name": "test_trigger",
                "movement_state": "combat" if i % 2 == 0 else "non_combat",
                "priority": 1,
                "word_count": 3,
                "voice_duration_ms": 1000
            }
            for i in range(10)
        ]

        result = await analyzer.analyze(stats)

        assert isinstance(result, AnalysisResult)
        # Should identify some patterns from the repeated phrases


class TestWeaknessAnalyzerPhrasePatterns:
    """Test phrase pattern analysis."""

    @pytest.mark.asyncio
    async def test_analyze_phrase_patterns(self):
        """Test phrase pattern detection."""
        analyzer = WeaknessAnalyzer()

        stats = SessionStatistics(session_dir="/test")
        stats.speeches = [
            {
                "timestamp_ms": 1000,
                "response": "Cover me, I need healing",
                "trigger_name": "low_hp",
                "movement_state": "combat"
            },
            {
                "timestamp_ms": 2000,
                "response": "Cover me, watch out",
                "trigger_name": "enemy",
                "movement_state": "combat"
            }
        ]

        patterns = await analyzer._analyze_phrase_patterns(stats)

        assert isinstance(patterns, list)
        # Should detect "Cover me" phrase

    @pytest.mark.asyncio
    async def test_analyze_vocabulary(self):
        """Test vocabulary analysis."""
        analyzer = WeaknessAnalyzer()

        stats = SessionStatistics(session_dir="/test")
        stats.total_word_count = 100
        stats.unique_vocabulary = {"hello", "world", "test", "game", "play"}
        stats.vocabulary_frequency = {
            "hello": 10,
            "world": 5,
            "test": 3,
            "game": 2,
            "play": 1
        }

        vocab_analysis = await analyzer._analyze_vocabulary(stats)

        assert vocab_analysis["unique_words"] == 5
        assert vocab_analysis["total_words"] == 100
        assert "diversity_score" in vocab_analysis

    @pytest.mark.asyncio
    async def test_identify_weaknesses(self):
        """Test weakness identification."""
        analyzer = WeaknessAnalyzer()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 2  # Low count
        stats.combat_speeches = 0
        stats.avg_response_time_ms = 6000  # Slow

        vocab_analysis = {"diversity_score": 0.3, "overused_words": []}

        weaknesses = await analyzer._identify_weaknesses(
            stats, vocab_analysis, []
        )

        assert len(weaknesses) > 0
        # Should identify low engagement and slow response

    @pytest.mark.asyncio
    async def test_identify_strengths(self):
        """Test strength identification."""
        analyzer = WeaknessAnalyzer()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 20  # Good engagement
        stats.avg_response_time_ms = 1500  # Fast
        stats.combat_speeches = 10
        stats.non_combat_speeches = 10

        vocab_analysis = {"diversity_score": 0.7}

        strengths = await analyzer._identify_strengths(stats, vocab_analysis)

        assert len(strengths) > 0
        # Should identify good engagement and response speed

    @pytest.mark.asyncio
    async def test_generate_suggestions(self):
        """Test suggestion generation."""
        analyzer = WeaknessAnalyzer()

        weaknesses = [
            WeaknessPoint(
                category="vocabulary",
                severity="minor",
                description="Low diversity",
                suggestion="Use more words"
            )
        ]

        vocab_analysis = {"diversity_score": 0.3}

        suggestions = await analyzer._generate_suggestions(weaknesses, vocab_analysis)

        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


# ============================================================================
# SpeechMetrics Tests
# ============================================================================

class TestSpeechMetrics:
    """Test SpeechMetrics dataclass."""

    def test_word_count_calculation(self):
        """Test word count is calculated in post_init."""
        metrics = SpeechMetrics(
            timestamp_ms=12345,
            trigger_name="test",
            response_text="Hello world test"
        )

        assert metrics.word_count == 3
        assert set(metrics.unique_words) == {"hello", "world", "test"}


# ============================================================================
# SessionStatistics Tests
# ============================================================================

class TestSessionStatistics:
    """Test SessionStatistics dataclass."""

    def test_init(self):
        """Test initialization."""
        stats = SessionStatistics(session_dir="/test/path")

        assert stats.session_dir == "/test/path"
        assert stats.total_speeches == 0
        assert stats.unique_vocabulary == set()

    def test_to_dict(self):
        """Test to_dict method."""
        stats = SessionStatistics(
            session_dir="/test/path",
            total_speeches=10,
            combat_speeches=5
        )

        stats_dict = stats.to_dict()

        assert stats_dict["session_dir"] == "/test/path"
        assert stats_dict["total_speeches"] == 10
        assert stats_dict["combat_speeches"] == 5

    def test_merge_vocabulary(self):
        """Test vocabulary merging."""
        stats = SessionStatistics(session_dir="/test")
        stats.merge_vocabulary(["hello", "world"])
        stats.merge_vocabulary(["hello", "test"])

        assert "hello" in stats.unique_vocabulary
        assert "world" in stats.unique_vocabulary
        assert "test" in stats.unique_vocabulary
        assert stats.vocabulary_frequency["hello"] == 2


# ============================================================================
# SessionStatsCollector Tests
# ============================================================================

class TestSessionStatsCollectorInit:
    """Test SessionStatsCollector initialization."""

    def test_init(self):
        """Test initialization."""
        collector = SessionStatsCollector("/test/path")

        assert collector.session_dir == Path("/test/path")
        assert collector.stats.session_dir == "/test/path"


class TestSessionStatsCollectorCollect:
    """Test SessionStatsCollector collect methods."""

    @pytest.mark.asyncio
    async def test_collect_no_logs(self):
        """Test collecting without log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SessionStatsCollector(tmpdir)
            stats = await collector.collect()

            assert isinstance(stats, SessionStatistics)
            assert stats.total_speeches == 0

    @pytest.mark.asyncio
    async def test_collect_with_responses(self):
        """Test collecting with response log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create response log
            response_log = tmpdir_path / "responses.jsonl"
            response_log.write_text(
                '{"timestamp_ms": 1000, "trigger_name": "test", "response": "Hello world", '
                '"movement_state": "combat", "priority": 1}\n'
                '{"timestamp_ms": 2000, "trigger_name": "test2", "response": "Test again", '
                '"movement_state": "non_combat", "priority": 2}\n'
            )

            collector = SessionStatsCollector(tmpdir)
            stats = await collector.collect()

            assert stats.total_speeches == 2
            assert stats.combat_speeches == 1
            assert stats.non_combat_speeches == 1

    @pytest.mark.asyncio
    async def test_process_response_entry(self):
        """Test processing individual response entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SessionStatsCollector(tmpdir)

            entry = {
                "timestamp_ms": 1000,
                "trigger_name": "test_trigger",
                "response": "Hello world test",
                "movement_state": "combat",
                "priority": 2,
                "voice_duration_ms": 1500
            }

            await collector._process_response_entry(entry)

            assert collector.stats.total_speeches == 1
            assert collector.stats.combat_speeches == 1
            assert "hello" in collector.stats.unique_vocabulary

    @pytest.mark.asyncio
    async def test_save(self):
        """Test saving statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = SessionStatsCollector(tmpdir)

            collector.stats.total_speeches = 10
            stats_path = await collector.save()

            assert Path(stats_path).exists()

            # Verify saved content
            with open(stats_path) as f:
                data = json.load(f)

            assert data["total_speeches"] == 10


class TestLoadStatistics:
    """Test load_statistics function."""

    @pytest.mark.asyncio
    async def test_load_statistics(self):
        """Test loading statistics from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"

            # Create test file
            test_data = {
                "session_dir": tmpdir,
                "total_speeches": 15,
                "combat_speeches": 8,
                "non_combat_speeches": 7,
                "vocabulary_frequency": {"hello": 5, "world": 3}
            }

            with open(stats_path, 'w') as f:
                json.dump(test_data, f)

            stats = await load_statistics(str(stats_path))

            assert stats.total_speeches == 15
            assert stats.combat_speeches == 8
            assert "hello" in stats.vocabulary_frequency


# ============================================================================
# CategoryScore Tests
# ============================================================================

class TestCategoryScore:
    """Test CategoryScore dataclass."""

    def test_init(self):
        """Test initialization."""
        score = CategoryScore(
            name="vocabulary",
            score=85.0,
            weight=0.25,
            details={"test": "value"}
        )

        assert score.name == "vocabulary"
        assert score.score == 85.0
        assert score.weight == 0.25

    def test_str(self):
        """Test string representation."""
        score = CategoryScore(
            name="pronunciation",
            score=75.5,
            weight=0.2,
            details={}
        )

        str_repr = str(score)
        assert "pronunciation" in str_repr
        assert "75.5" in str_repr


# ============================================================================
# OverallScore Tests
# ============================================================================

class TestOverallScore:
    """Test OverallScore dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        categories = [
            CategoryScore(
                name="pronunciation",
                score=85.0,
                weight=0.2,
                details={"test": "value"}
            ),
            CategoryScore(
                name="vocabulary",
                score=90.0,
                weight=0.25,
                details={}
            )
        ]

        score = OverallScore(
            total_score=87.5,
            categories=categories,
            grade="B",
            summary="Good performance"
        )

        score_dict = score.to_dict()

        assert score_dict["total_score"] == 87.5
        assert score_dict["grade"] == "B"
        assert len(score_dict["categories"]) == 2


# ============================================================================
# ScoreCalculator Tests
# ============================================================================

class TestScoreCalculatorInit:
    """Test ScoreCalculator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        calculator = ScoreCalculator()

        assert calculator.CATEGORY_WEIGHTS["pronunciation"] == 0.20
        assert calculator.CATEGORY_WEIGHTS["vocabulary"] == 0.25

    def test_init_custom_weights(self):
        """Test custom category weights."""
        custom_weights = {
            "pronunciation": 0.3,
            "vocabulary": 0.3,
            "response_speed": 0.2,
            "strategic_thinking": 0.2
        }

        calculator = ScoreCalculator(custom_weights=custom_weights)

        assert calculator.CATEGORY_WEIGHTS == custom_weights


class TestScoreCalculatorCalculate:
    """Test ScoreCalculator calculate methods."""

    @pytest.mark.asyncio
    async def test_calculate_empty_session(self):
        """Test calculating score for empty session."""
        calculator = ScoreCalculator()

        stats = SessionStatistics(session_dir="/test")
        score = await calculator.calculate(stats)

        assert isinstance(score, OverallScore)
        assert 0 <= score.total_score <= 100
        assert score.grade in ["A", "B", "C", "D", "F"]

    @pytest.mark.asyncio
    async def test_calculate_with_data(self):
        """Test calculating score with data."""
        calculator = ScoreCalculator()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 20
        stats.combat_speeches = 10
        stats.non_combat_speeches = 10
        stats.avg_response_time_ms = 2000
        stats.total_word_count = 200
        stats.unique_vocabulary = {f"word{i}" for i in range(50)}
        stats.vocabulary_frequency = {f"word{i}": i+1 for i in range(50)}
        stats.speeches = [
            {
                "timestamp_ms": i * 10000,
                "response": "test response",
                "voice_duration_ms": 1500
            }
            for i in range(20)
        ]
        stats.response_times = [1500, 2000, 2500, 1800, 2200]

        score = await calculator.calculate(stats)

        assert isinstance(score, OverallScore)
        # With good data, should have decent score
        assert score.total_score >= 50

    @pytest.mark.asyncio
    async def test_calculate_pronunciation_score(self):
        """Test pronunciation score calculation."""
        calculator = ScoreCalculator()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 10
        stats.avg_voice_duration_ms = 1500
        stats.speeches = [
            {"voice_duration_ms": 1500} for _ in range(10)
        ]

        category_score = await calculator._calculate_pronunciation_score(stats)

        assert isinstance(category_score, CategoryScore)
        assert category_score.name == "pronunciation"
        assert 0 <= category_score.score <= 100

    @pytest.mark.asyncio
    async def test_calculate_vocabulary_score(self):
        """Test vocabulary score calculation."""
        calculator = ScoreCalculator()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 10
        stats.total_word_count = 100
        stats.unique_vocabulary = {f"word{i}" for i in range(40)}
        stats.vocabulary_frequency = {f"word{i}": i+1 for i in range(40)}

        category_score = await calculator._calculate_vocabulary_score(stats)

        assert isinstance(category_score, CategoryScore)
        assert category_score.name == "vocabulary"
        assert 0 <= category_score.score <= 100

    @pytest.mark.asyncio
    async def test_calculate_response_speed_score(self):
        """Test response speed score calculation."""
        calculator = ScoreCalculator()

        stats = SessionStatistics(session_dir="/test")
        stats.avg_response_time_ms = 2000
        stats.min_response_time_ms = 1000
        stats.max_response_time_ms = 3000
        stats.response_times = [1500, 2000, 2500, 1800, 2200]

        category_score = await calculator._calculate_response_speed_score(stats)

        assert isinstance(category_score, CategoryScore)
        assert category_score.name == "response_speed"
        assert 0 <= category_score.score <= 100

    @pytest.mark.asyncio
    async def test_calculate_strategic_thinking_score(self):
        """Test strategic thinking score calculation."""
        calculator = ScoreCalculator()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 15
        stats.combat_speeches = 8
        stats.non_combat_speeches = 7
        stats.high_priority_count = 5
        stats.medium_priority_count = 5
        stats.low_priority_count = 5
        stats.trigger_counts = {
            f"trigger{i}": i+1 for i in range(10)
        }

        category_score = await calculator._calculate_strategic_thinking_score(stats)

        assert isinstance(category_score, CategoryScore)
        assert category_score.name == "strategic_thinking"
        assert 0 <= category_score.score <= 100

    def test_determine_grade(self):
        """Test grade determination."""
        calculator = ScoreCalculator()

        assert calculator._determine_grade(95) == "A"
        assert calculator._determine_grade(85) == "B"
        assert calculator._determine_grade(65) == "C"
        assert calculator._determine_grade(45) == "D"
        assert calculator._determine_grade(20) == "F"

    def test_generate_summary(self):
        """Test summary generation."""
        calculator = ScoreCalculator()

        categories = [
            CategoryScore(
                name="vocabulary",
                score=90.0,
                weight=0.25,
                details={}
            ),
            CategoryScore(
                name="response_speed",
                score=60.0,
                weight=0.20,
                details={}
            )
        ]

        summary = calculator._generate_summary(75.0, categories)

        assert isinstance(summary, str)
        assert "vocabulary" in summary.lower() or "response_speed" in summary.lower()


# ============================================================================
# ReviewReportGenerator Tests
# ============================================================================

class TestReviewReportGenerator:
    """Test ReviewReportGenerator class."""

    def test_init(self):
        """Test initialization."""
        generator = ReviewReportGenerator()

        assert generator.logger is not None

    @pytest.mark.asyncio
    async def test_generate_text_report(self):
        """Test generating text report."""
        generator = ReviewReportGenerator()

        stats = SessionStatistics(session_dir="/test")
        stats.total_speeches = 10

        score = OverallScore(
            total_score=75.0,
            categories=[],
            grade="B",
            summary="Good performance"
        )

        analysis = AnalysisResult(
            weaknesses=[],
            strengths=[],
            phrase_patterns=[],
            vocabulary_analysis={
                "unique_words": 50,
                "total_words": 100,
                "diversity_score": 0.5,
                "avg_word_length": 4.5
            },
            improvement_suggestions=["Practice more"]
        )

        text = await generator.generate(stats, score, analysis, "text")

        assert "text" in text
        assert isinstance(text["text"], str)

    @pytest.mark.asyncio
    async def test_generate_json_report(self):
        """Test generating JSON report."""
        generator = ReviewReportGenerator()

        stats = SessionStatistics(session_dir="/test")
        score = OverallScore(
            total_score=80.0,
            categories=[],
            grade="A",
            summary="Excellent"
        )
        analysis = AnalysisResult(
            weaknesses=[],
            strengths=[],
            phrase_patterns=[],
            vocabulary_analysis={},
            improvement_suggestions=[]
        )

        result = await generator.generate(stats, score, analysis, "json")

        assert "json" in result

        # Verify it's valid JSON
        data = json.loads(result["json"])
        assert "score" in data
        assert "statistics" in data

    @pytest.mark.asyncio
    async def test_generate_both_formats(self):
        """Test generating both text and JSON reports."""
        generator = ReviewReportGenerator()

        stats = SessionStatistics(session_dir="/test")
        score = OverallScore(
            total_score=70.0,
            categories=[],
            grade="C",
            summary="Fair"
        )
        analysis = AnalysisResult(
            weaknesses=[],
            strengths=[],
            phrase_patterns=[],
            vocabulary_analysis={
                "unique_words": 30,
                "total_words": 60,
                "diversity_score": 0.5,
                "avg_word_length": 4.0
            },
            improvement_suggestions=[]
        )

        result = await generator.generate(stats, score, analysis, "both")

        assert "text" in result
        assert "json" in result


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestAnalyzeSession:
    """Test analyze_session helper."""

    @pytest.mark.asyncio
    async def test_analyze_session(self):
        """Test analyzing a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty response log
            response_log = Path(tmpdir) / "responses.jsonl"
            response_log.write_text('')

            result = await analyze_session(tmpdir)

            assert isinstance(result, AnalysisResult)


# ============================================================================
# Integration Tests
# ============================================================================

class TestReviewIntegration:
    """Integration tests for review functionality."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """Test complete analysis flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            response_log = Path(tmpdir) / "responses.jsonl"
            response_log.write_text(
                '{"timestamp_ms": 1000, "trigger_name": "low_hp", "response": "Low HP, need healing", '
                '"movement_state": "combat", "priority": 3, "voice_duration_ms": 1500}\n'
                '{"timestamp_ms": 2000, "trigger_name": "enemy", "response": "Enemy detected, cover me", '
                '"movement_state": "combat", "priority": 2, "voice_duration_ms": 1000}\n'
                '{"timestamp_ms": 3000, "trigger_name": "loot", "response": "Loot the area", '
                '"movement_state": "non_combat", "priority": 1, "voice_duration_ms": 800}\n'
            )

            # Collect statistics
            collector = SessionStatsCollector(tmpdir)
            stats = await collector.collect()

            # Calculate score
            calculator = ScoreCalculator()
            score = await calculator.calculate(stats)

            # Analyze weaknesses
            analyzer = WeaknessAnalyzer()
            analysis = await analyzer.analyze(stats)

            # Generate report
            generator = ReviewReportGenerator()
            report = await generator.generate(stats, score, analysis, "text")

            assert stats.total_speeches == 3
            assert score.total_score >= 0
            assert isinstance(analysis, AnalysisResult)
            assert "text" in report
