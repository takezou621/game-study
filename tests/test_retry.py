#!/usr/bin/env python3
"""Tests for retry functionality (synchronous only)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.retry import retry_with_backoff_sync
from utils.exceptions import APIError, RateLimitError


def test_retry_success_first_try():
    """Test retry function succeeds on first try."""
    call_count = []

    def successful_func():
        call_count.append(1)
        return "success"

    result = retry_with_backoff_sync(successful_func, max_retries=3)

    assert result == "success"
    assert len(call_count) == 1


def test_retry_success_after_retries():
    """Test retry function succeeds after a few attempts."""
    call_count = []

    def eventually_successful():
        call_count.append(1)
        if len(call_count) < 3:
            raise APIError("Temporary error")
        return "success"

    result = retry_with_backoff_sync(eventually_successful, max_retries=3)

    assert result == "success"
    assert len(call_count) == 3


def test_retry_exhausted():
    """Test retry function gives up after max retries."""
    call_count = []

    def always_fail():
        call_count.append(1)
        raise APIError("Always fails")

    with pytest.raises(APIError, match="Always fails"):
        retry_with_backoff_sync(always_fail, max_retries=2)

    assert len(call_count) == 3  # Initial + 2 retries


def test_retry_non_retryable_exception():
    """Test that non-retryable exceptions are raised immediately."""
    call_count = []

    def raise_value_error():
        call_count.append(1)
        raise ValueError("Not retryable")

    with pytest.raises(ValueError, match="Not retryable"):
        retry_with_backoff_sync(raise_value_error, max_retries=3)

    assert len(call_count) == 1  # Should not retry


def test_retry_rate_limit_with_retry_after():
    """Test that RateLimitError uses retry_after for delay."""
    call_count = []
    delays = []

    def on_retry_callback(attempt, error, delay):
        delays.append(delay)

    def raise_rate_limit():
        call_count.append(1)
        if len(call_count) < 2:
            raise RateLimitError("Rate limit", retry_after=0.1)
        return "success"

    result = retry_with_backoff_sync(
        raise_rate_limit,
        max_retries=3,
        on_retry=on_retry_callback
    )

    assert result == "success"
    assert len(call_count) == 2
    assert len(delays) == 1
    assert delays[0] == 0.1  # Should use retry_after


def test_retry_max_delay():
    """Test that delay is capped at max_delay."""
    import time

    call_count = []

    def always_fail():
        call_count.append(1)
        raise APIError("Always fails")

    start_time = time.time()

    with pytest.raises(APIError):
        retry_with_backoff_sync(
            always_fail,
            max_retries=5,
            base_delay=10.0,
            max_delay=0.5
        )

    elapsed = time.time() - start_time

    # With exponential backoff and 5 retries, if uncapped would be much longer
    # With max_delay=0.5, should be approximately 0.5 * 5 = 2.5s
    assert elapsed < 4.0


def test_retry_custom_exceptions():
    """Test retry with custom retryable exceptions."""
    class CustomError(Exception):
        pass

    call_count = []

    def raise_custom():
        call_count.append(1)
        if len(call_count) < 2:
            raise CustomError("Custom error")
        return "success"

    result = retry_with_backoff_sync(
        raise_custom,
        max_retries=3,
        retryable_exceptions=(CustomError,)
    )

    assert result == "success"
    assert len(call_count) == 2


def test_retry_zero_retries():
    """Test retry with max_retries=0 (no retries)."""
    call_count = []

    def always_fail():
        call_count.append(1)
        raise APIError("Always fails")

    with pytest.raises(APIError):
        retry_with_backoff_sync(always_fail, max_retries=0)

    assert len(call_count) == 1  # Only called once, no retries
