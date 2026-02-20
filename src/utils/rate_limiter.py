"""Rate limiting utilities for API calls."""

import time
import logging
from typing import Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.

    Limits the number of calls within a time period using a sliding window algorithm.

    Example:
        limiter = RateLimiter(max_calls=10, period_seconds=60)
        if limiter.allow_call():
            make_api_call()
        else:
            wait_time = limiter.wait_time()
            time.sleep(wait_time)
            make_api_call()
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        name: Optional[str] = None
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period_seconds: Time period in seconds
            name: Optional name for logging (e.g., "OpenAI API")
        """
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")

        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.name = name or "RateLimiter"

        # Use a deque to store timestamps of recent calls
        self._call_timestamps: deque[float] = deque()
        self._lock = threading.Lock()

        logger.debug(
            f"{self.name} initialized: {max_calls} calls per {period_seconds} seconds"
        )

    def allow_call(self) -> bool:
        """
        Check if a call is allowed under the rate limit.

        Returns:
            True if call is allowed, False otherwise
        """
        with self._lock:
            now = time.time()

            # Remove timestamps older than the period
            while self._call_timestamps:
                if now - self._call_timestamps[0] >= self.period_seconds:
                    self._call_timestamps.popleft()
                else:
                    break

            # Check if we can make a call
            if len(self._call_timestamps) < self.max_calls:
                self._call_timestamps.append(now)
                logger.debug(
                    f"{self.name}: Call allowed ({len(self._call_timestamps)}/{self.max_calls})"
                )
                return True

            logger.warning(
                f"{self.name}: Rate limit exceeded ({len(self._call_timestamps)}/{self.max_calls})"
            )
            return False

    def wait_time(self) -> float:
        """
        Calculate time to wait before next allowed call.

        Returns:
            Time in seconds to wait, or 0 if call is allowed now
        """
        with self._lock:
            now = time.time()

            # Remove old timestamps
            while self._call_timestamps:
                if now - self._call_timestamps[0] >= self.period_seconds:
                    self._call_timestamps.popleft()
                else:
                    break

            # If under limit, no wait needed
            if len(self._call_timestamps) < self.max_calls:
                return 0.0

            # Calculate wait time based on oldest call in window
            oldest_call = self._call_timestamps[0]
            wait_time = self.period_seconds - (now - oldest_call)

            return max(0.0, wait_time)

    def reset(self) -> None:
        """Reset the rate limiter (clear all call history)."""
        with self._lock:
            self._call_timestamps.clear()
            logger.debug(f"{self.name}: Rate limiter reset")

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with current usage statistics
        """
        with self._lock:
            now = time.time()

            # Clean up old timestamps first
            while self._call_timestamps:
                if now - self._call_timestamps[0] >= self.period_seconds:
                    self._call_timestamps.popleft()
                else:
                    break

            return {
                "name": self.name,
                "max_calls": self.max_calls,
                "period_seconds": self.period_seconds,
                "current_calls": len(self._call_timestamps),
                "remaining_calls": max(0, self.max_calls - len(self._call_timestamps)),
                "wait_time": self.wait_time(),
            }
