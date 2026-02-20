"""Rate limiter for API calls.

Implements a Token Bucket algorithm to control the rate of API calls.
"""

import time
import threading
from typing import Optional


class RateLimiter:
    """
    Token Bucket rate limiter.

    Allows a maximum number of calls within a time period,
    with support for bursting up to the bucket capacity.
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        bucket_capacity: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period_seconds: Time period in seconds
            bucket_capacity: Maximum bucket capacity (defaults to max_calls)
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.bucket_capacity = bucket_capacity or max_calls

        # Token bucket state
        self._tokens = float(self.bucket_capacity)
        self._last_update = time.monotonic()

        # Rate of token replenishment (tokens per second)
        self._refill_rate = max_calls / period_seconds

        # Thread safety
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update

        # Calculate tokens to add
        tokens_to_add = elapsed * self._refill_rate
        self._tokens = min(self.bucket_capacity, self._tokens + tokens_to_add)
        self._last_update = now

    def allow_call(self) -> bool:
        """
        Check if a call is allowed.

        Returns:
            True if the call is allowed, False if rate limit exceeded
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            return False

    def wait_for_token(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until a token is available.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.monotonic()

        while True:
            if self.allow_call():
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            # Calculate wait time for next token
            with self._lock:
                self._refill()
                tokens_needed = 1.0 - self._tokens
                wait_time = tokens_needed / self._refill_rate

            # Sleep for a short time before retrying
            time.sleep(min(wait_time, 0.1))

    def get_wait_time(self) -> float:
        """
        Get estimated wait time until next token is available.

        Returns:
            Estimated wait time in seconds (0 if token available now)
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                return 0.0

            tokens_needed = 1.0 - self._tokens
            return tokens_needed / self._refill_rate

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        with self._lock:
            self._tokens = float(self.bucket_capacity)
            self._last_update = time.monotonic()


class AsyncRateLimiter:
    """
    Async-compatible Token Bucket rate limiter.
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        bucket_capacity: Optional[int] = None
    ):
        """
        Initialize async rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period_seconds: Time period in seconds
            bucket_capacity: Maximum bucket capacity (defaults to max_calls)
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.bucket_capacity = bucket_capacity or max_calls

        # Token bucket state
        self._tokens = float(self.bucket_capacity)
        self._last_update = time.monotonic()

        # Rate of token replenishment
        self._refill_rate = max_calls / period_seconds

        # Thread safety (for sync operations in async context)
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update

        tokens_to_add = elapsed * self._refill_rate
        self._tokens = min(self.bucket_capacity, self._tokens + tokens_to_add)
        self._last_update = now

    def allow_call(self) -> bool:
        """
        Check if a call is allowed (non-blocking).

        Returns:
            True if the call is allowed, False if rate limit exceeded
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            return False

    async def wait_for_token(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until a token is available (async).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if token acquired, False if timeout
        """
        import asyncio

        start_time = time.monotonic()

        while True:
            if self.allow_call():
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            # Calculate wait time
            wait_time = self.get_wait_time()
            await asyncio.sleep(min(wait_time, 0.1))

    def get_wait_time(self) -> float:
        """
        Get estimated wait time until next token is available.

        Returns:
            Estimated wait time in seconds (0 if token available now)
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                return 0.0

            tokens_needed = 1.0 - self._tokens
            return tokens_needed / self._refill_rate

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        with self._lock:
            self._tokens = float(self.bucket_capacity)
            self._last_update = time.monotonic()
