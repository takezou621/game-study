"""Metrics collection for monitoring.

This module provides thread-safe metrics collection for monitoring
application performance and behavior. It supports counter metrics
and latency sampling with statistical summaries.
"""

from collections import defaultdict
from math import ceil
from threading import Lock
from typing import Any


class MetricsCollector:
    """Thread-safe metrics collector for monitoring application performance.

    Tracks counter metrics (incrementing values) and latency metrics
    (timing samples) with thread-safe operations.

    Attributes:
        _lock: Thread lock for safe concurrent access
        _counters: Dictionary tracking counter values by metric name
        _latency_samples: List of recorded latency samples in seconds
        _max_samples: Maximum number of latency samples to retain
    """

    def __init__(self, max_samples: int = 1000):
        """Initialize the metrics collector.

        Args:
            max_samples: Maximum number of latency samples to retain.
                Defaults to 1000. When exceeded, oldest samples are removed.
        """
        self._lock = Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latency_samples: list[float] = []
        self._max_samples = max_samples

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric.

        Thread-safe increment of a named counter.

        Args:
            metric: Name of the metric to increment
            value: Amount to increment by (default: 1)
        """
        with self._lock:
            self._counters[metric] += value

    def decrement(self, metric: str, value: int = 1) -> None:
        """Decrement a counter metric.

        Thread-safe decrement of a named counter.

        Args:
            metric: Name of the metric to decrement
            value: Amount to decrement by (default: 1)
        """
        with self._lock:
            self._counters[metric] -= value

    def record_latency(self, seconds: float) -> None:
        """Record a latency sample.

        Thread-safe recording of a latency measurement. Maintains
        a maximum number of samples by removing oldest when limit is reached.

        Args:
            seconds: Latency value in seconds
        """
        with self._lock:
            self._latency_samples.append(seconds)

            # Trim if exceeding max samples
            if len(self._latency_samples) > self._max_samples:
                # Remove oldest samples (from the beginning)
                excess = len(self._latency_samples) - self._max_samples
                self._latency_samples = self._latency_samples[excess:]

    def get_counter(self, metric: str) -> int:
        """Get the current value of a counter metric.

        Args:
            metric: Name of the counter metric

        Returns:
            Current counter value (0 if metric doesn't exist)
        """
        with self._lock:
            return self._counters.get(metric, 0)

    def reset_counter(self, metric: str) -> int:
        """Reset a counter metric to zero.

        Args:
            metric: Name of the counter metric

        Returns:
            The previous value before reset
        """
        with self._lock:
            previous = self._counters.get(metric, 0)
            self._counters[metric] = 0
            return previous

    def reset_all_counters(self) -> dict[str, int]:
        """Reset all counter metrics to zero.

        Returns:
            Dictionary of previous values before reset
        """
        with self._lock:
            previous = dict(self._counters)
            self._counters.clear()
            return previous

    def get_latency_samples(self) -> list[float]:
        """Get all current latency samples.

        Returns:
            Copy of the latency samples list
        """
        with self._lock:
            return list(self._latency_samples)

    def clear_latency_samples(self) -> list[float]:
        """Clear all latency samples.

        Returns:
            The previous samples before clearing
        """
        with self._lock:
            previous = list(self._latency_samples)
            self._latency_samples.clear()
            return previous

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary including counts and latency statistics.

        Calculates statistical measures for latency samples including
        minimum, maximum, average, and 95th percentile.

        Returns:
            Dictionary with:
                - counters: Dictionary of all counter values
                - latency_count: Number of latency samples
                - latency_min: Minimum latency in seconds
                - latency_max: Maximum latency in seconds
                - latency_avg: Average latency in seconds
                - latency_p95: 95th percentile latency in seconds
        """
        with self._lock:
            summary: dict[str, Any] = {
                "counters": dict(self._counters),
                "latency_count": len(self._latency_samples),
            }

            if self._latency_samples:
                sorted_samples = sorted(self._latency_samples)
                summary["latency_min"] = sorted_samples[0]
                summary["latency_max"] = sorted_samples[-1]
                summary["latency_avg"] = sum(sorted_samples) / len(sorted_samples)

                # Calculate 95th percentile
                p95_index = ceil(len(sorted_samples) * 0.95) - 1
                summary["latency_p95"] = sorted_samples[max(0, p95_index)]
            else:
                summary["latency_min"] = 0.0
                summary["latency_max"] = 0.0
                summary["latency_avg"] = 0.0
                summary["latency_p95"] = 0.0

            return summary


# Global metrics instance
metrics = MetricsCollector()


__all__ = [
    "MetricsCollector",
    "metrics",
]
