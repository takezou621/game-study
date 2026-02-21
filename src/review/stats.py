"""Session statistics collection and storage for review functionality."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeechMetrics:
    """Metrics for a single speech/response event."""

    timestamp_ms: int
    trigger_name: str
    response_text: str
    voice_duration_ms: Optional[int] = None
    movement_state: str = "non_combat"
    priority: int = 0
    word_count: int = 0
    unique_words: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate word count and unique words after initialization."""
        if self.response_text:
            words = self.response_text.lower().split()
            self.word_count = len(words)
            self.unique_words = list(set(words))


@dataclass
class SessionStatistics:
    """Collected statistics for a gaming session."""

    session_dir: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    # Speech metrics
    total_speeches: int = 0
    combat_speeches: int = 0
    non_combat_speeches: int = 0

    # Response time metrics (milliseconds)
    avg_response_time_ms: float = 0.0
    min_response_time_ms: Optional[int] = None
    max_response_time_ms: Optional[int] = None
    response_times: List[int] = field(default_factory=list)

    # Voice metrics
    total_voice_duration_ms: int = 0
    avg_voice_duration_ms: float = 0.0

    # Vocabulary metrics
    total_word_count: int = 0
    unique_vocabulary: set = field(default_factory=set)
    vocabulary_frequency: Dict[str, int] = field(default_factory=dict)

    # Trigger distribution
    trigger_counts: Dict[str, int] = field(default_factory=dict)

    # Priority distribution
    high_priority_count: int = 0
    medium_priority_count: int = 0
    low_priority_count: int = 0

    # Individual speech events for detailed analysis
    speeches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_dir": self.session_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_speeches": self.total_speeches,
            "combat_speeches": self.combat_speeches,
            "non_combat_speeches": self.non_combat_speeches,
            "avg_response_time_ms": self.avg_response_time_ms,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "total_voice_duration_ms": self.total_voice_duration_ms,
            "avg_voice_duration_ms": self.avg_voice_duration_ms,
            "total_word_count": self.total_word_count,
            "unique_vocabulary_size": len(self.unique_vocabulary),
            "vocabulary_frequency": self.vocabulary_frequency,
            "trigger_counts": self.trigger_counts,
            "high_priority_count": self.high_priority_count,
            "medium_priority_count": self.medium_priority_count,
            "low_priority_count": self.low_priority_count,
            "speeches": self.speeches,
        }

    def merge_vocabulary(self, words: List[str]) -> None:
        """Merge new words into vocabulary statistics."""
        for word in words:
            self.unique_vocabulary.add(word)
            self.vocabulary_frequency[word] = self.vocabulary_frequency.get(word, 0) + 1


class SessionStatsCollector:
    """Collector for session statistics from SessionLogger data."""

    def __init__(self, session_dir: str):
        """
        Initialize statistics collector.

        Args:
            session_dir: Directory containing session logs
        """
        self.session_dir = Path(session_dir)
        self.stats = SessionStatistics(session_dir=session_dir)
        self.logger = logger

    async def collect(self) -> SessionStatistics:
        """
        Collect statistics from session log files.

        Returns:
            Collected session statistics
        """
        try:
            # Load and process response logs
            await self._process_response_logs()

            # Load and process trigger logs
            await self._process_trigger_logs()

            # Calculate derived statistics
            self._calculate_derived_stats()

            # Set end time
            self.stats.end_time = datetime.now().isoformat()

            self.logger.info(f"Collected statistics: {self.stats.total_speeches} speeches")
            return self.stats

        except Exception as e:
            self.logger.error(f"Error collecting statistics: {e}", e)
            raise

    async def _process_response_logs(self) -> None:
        """Process response.jsonl file to extract speech metrics."""
        response_log_path = self.session_dir / "responses.jsonl"

        if not response_log_path.exists():
            self.logger.warning(f"Response log not found: {response_log_path}")
            return

        try:
            with open(response_log_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        response_data = json.loads(line)
                        await self._process_response_entry(response_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing response log line: {e}")

        except IOError as e:
            self.logger.error(f"Error reading response log: {e}", e)

    async def _process_response_entry(self, entry: Dict[str, Any]) -> None:
        """Process a single response entry."""
        timestamp_ms = entry.get("timestamp_ms", 0)
        trigger_name = entry.get("trigger_name", "unknown")
        response_text = entry.get("response", "")
        movement_state = entry.get("movement_state", "non_combat")
        priority = entry.get("priority", 0)
        voice_duration_ms = entry.get("voice_duration_ms")

        # Create speech metrics
        speech = SpeechMetrics(
            timestamp_ms=timestamp_ms,
            trigger_name=trigger_name,
            response_text=response_text,
            voice_duration_ms=voice_duration_ms,
            movement_state=movement_state,
            priority=priority
        )

        # Update basic counts
        self.stats.total_speeches += 1
        if movement_state == "combat":
            self.stats.combat_speeches += 1
        else:
            self.stats.non_combat_speeches += 1

        # Update priority counts
        if priority >= 3:
            self.stats.high_priority_count += 1
        elif priority >= 2:
            self.stats.medium_priority_count += 1
        else:
            self.stats.low_priority_count += 1

        # Update trigger counts
        self.stats.trigger_counts[trigger_name] = \
            self.stats.trigger_counts.get(trigger_name, 0) + 1

        # Update vocabulary
        self.stats.total_word_count += speech.word_count
        self.stats.merge_vocabulary(speech.unique_words)

        # Update voice duration
        if voice_duration_ms:
            self.stats.total_voice_duration_ms += voice_duration_ms

        # Store speech event for detailed analysis
        self.stats.speeches.append({
            "timestamp_ms": timestamp_ms,
            "trigger_name": trigger_name,
            "response": response_text,
            "movement_state": movement_state,
            "priority": priority,
            "word_count": speech.word_count,
            "voice_duration_ms": voice_duration_ms,
        })

    async def _process_trigger_logs(self) -> None:
        """Process triggers.jsonl file for additional context."""
        trigger_log_path = self.session_dir / "triggers.jsonl"

        if not trigger_log_path.exists():
            return

        try:
            with open(trigger_log_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        trigger_data = json.loads(line)
                        # Extract timing information if available
                        # Response time can be calculated from trigger to response
                    except json.JSONDecodeError:
                        pass

        except IOError as e:
            self.logger.error(f"Error reading trigger log: {e}", e)

    def _calculate_derived_stats(self) -> None:
        """Calculate derived statistics from collected data."""
        # Voice duration average
        if self.stats.total_speeches > 0:
            voice_count = sum(
                1 for s in self.stats.speeches
                if s.get("voice_duration_ms") is not None
            )
            if voice_count > 0:
                self.stats.avg_voice_duration_ms = \
                    self.stats.total_voice_duration_ms / voice_count

        # Response time statistics
        response_times = []
        for i, speech in enumerate(self.stats.speeches):
            if i > 0:
                prev_speech = self.stats.speeches[i - 1]
                time_diff = speech["timestamp_ms"] - prev_speech["timestamp_ms"]
                # Only count reasonable response times (less than 5 minutes)
                if 0 < time_diff < 300000:
                    response_times.append(time_diff)

        if response_times:
            self.stats.response_times = response_times
            self.stats.avg_response_time_ms = sum(response_times) / len(response_times)
            self.stats.min_response_time_ms = min(response_times)
            self.stats.max_response_time_ms = max(response_times)

    async def save(self, output_path: Optional[str] = None) -> str:
        """
        Save statistics to JSON file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.session_dir / "statistics.json"
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.info(f"Statistics saved to: {output_path}")
            return str(output_path)

        except IOError as e:
            self.logger.error(f"Error saving statistics: {e}", e)
            raise


async def load_statistics(stats_path: str) -> SessionStatistics:
    """
    Load statistics from JSON file.

    Args:
        stats_path: Path to statistics JSON file

    Returns:
        Loaded SessionStatistics object
    """
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)

        # Reconstruct SessionStatistics from dict
        stats = SessionStatistics(session_dir=data["session_dir"])
        stats.start_time = data.get("start_time")
        stats.end_time = data.get("end_time")
        stats.total_speeches = data.get("total_speeches", 0)
        stats.combat_speeches = data.get("combat_speeches", 0)
        stats.non_combat_speeches = data.get("non_combat_speeches", 0)
        stats.avg_response_time_ms = data.get("avg_response_time_ms", 0.0)
        stats.min_response_time_ms = data.get("min_response_time_ms")
        stats.max_response_time_ms = data.get("max_response_time_ms")
        stats.total_voice_duration_ms = data.get("total_voice_duration_ms", 0)
        stats.avg_voice_duration_ms = data.get("avg_voice_duration_ms", 0.0)
        stats.total_word_count = data.get("total_word_count", 0)
        stats.unique_vocabulary = set()
        stats.vocabulary_frequency = data.get("vocabulary_frequency", {})
        stats.trigger_counts = data.get("trigger_counts", {})
        stats.high_priority_count = data.get("high_priority_count", 0)
        stats.medium_priority_count = data.get("medium_priority_count", 0)
        stats.low_priority_count = data.get("low_priority_count", 0)
        stats.speeches = data.get("speeches", [])

        # Reconstruct unique vocabulary from frequency map
        if stats.vocabulary_frequency:
            stats.unique_vocabulary = set(stats.vocabulary_frequency.keys())

        return stats

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading statistics from {stats_path}: {e}")
        raise
