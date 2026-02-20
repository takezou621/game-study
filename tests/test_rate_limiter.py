#!/usr/bin/env python3
"""Tests for RateLimiter."""

import sys
from pathlib import Path
import time
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.rate_limiter import RateLimiter


def test_rate_limiter_initialization():
    """Test RateLimiter initialization."""
    limiter = RateLimiter(max_calls=10, period_seconds=60)

    assert limiter is not None
    assert limiter.max_calls == 10
    assert limiter.period_seconds == 60.0


def test_rate_limiter_allow_call_under_limit():
    """Test allow_call when under limit."""
    limiter = RateLimiter(max_calls=5, period_seconds=60)

    # Should allow first 5 calls
    for _ in range(5):
        assert limiter.allow_call() is True


def test_rate_limiter_block_over_limit():
    """Test allow_call when over limit."""
    limiter = RateLimiter(max_calls=3, period_seconds=60)

    # Should allow first 3 calls
    for _ in range(3):
        assert limiter.allow_call() is True

    # Should block the 4th call
    assert limiter.allow_call() is False


def test_rate_limiter_sliding_window():
    """Test that the sliding window correctly expires old calls."""
    limiter = RateLimiter(max_calls=2, period_seconds=1)

    # Allow first 2 calls
    assert limiter.allow_call() is True
    assert limiter.allow_call() is True

    # Should block 3rd call
    assert limiter.allow_call() is False

    # Wait for the period to expire
    time.sleep(1.1)

    # Should allow call again
    assert limiter.allow_call() is True


def test_rate_limiter_wait_time():
    """Test wait_time calculation."""
    limiter = RateLimiter(max_calls=2, period_seconds=1)

    # No wait time when under limit
    assert limiter.wait_time() == 0.0

    # Make calls up to limit
    limiter.allow_call()
    limiter.allow_call()

    # Wait time should be positive when over limit
    wait_time = limiter.wait_time()
    assert wait_time > 0.0
    assert wait_time <= 1.0


def test_rate_limiter_reset():
    """Test resetting the rate limiter."""
    limiter = RateLimiter(max_calls=2, period_seconds=60)

    # Make calls up to limit
    limiter.allow_call()
    limiter.allow_call()

    # Should be blocked
    assert limiter.allow_call() is False

    # Reset
    limiter.reset()

    # Should allow calls again
    assert limiter.allow_call() is True


def test_rate_limiter_get_stats():
    """Test getting rate limiter statistics."""
    limiter = RateLimiter(max_calls=5, period_seconds=60, name="TestLimiter")

    stats = limiter.get_stats()

    assert stats["name"] == "TestLimiter"
    assert stats["max_calls"] == 5
    assert stats["period_seconds"] == 60.0
    assert stats["current_calls"] == 0
    assert stats["remaining_calls"] == 5

    # Make some calls
    limiter.allow_call()
    limiter.allow_call()

    stats = limiter.get_stats()
    assert stats["current_calls"] == 2
    assert stats["remaining_calls"] == 3


def test_rate_limiter_invalid_parameters():
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="max_calls must be positive"):
        RateLimiter(max_calls=0, period_seconds=60)

    with pytest.raises(ValueError, match="period_seconds must be positive"):
        RateLimiter(max_calls=10, period_seconds=0)


def test_rate_limiter_thread_safety():
    """Test that RateLimiter is thread-safe."""
    import threading

    limiter = RateLimiter(max_calls=100, period_seconds=60)
    results = []

    def make_calls(n):
        for _ in range(n):
            results.append(limiter.allow_call())

    # Create multiple threads
    threads = [
        threading.Thread(target=make_calls, args=(25,))
        for _ in range(4)
    ]

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Should have exactly 100 successful calls
    assert sum(results) == 100
    assert len(results) == 100
