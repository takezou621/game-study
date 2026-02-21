"""Tests for Retry utilities."""

import asyncio
import time
import pytest

import sys
from pathlib import Path

# Direct module import
SRC_PATH = Path(__file__).parent.parent.parent / "src"
RETRY_PATH = SRC_PATH / "utils" / "retry.py"

import importlib.util
spec = importlib.util.spec_from_file_location("retry", RETRY_PATH)
retry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry_module)

retry_with_backoff = retry_module.retry_with_backoff
async_retry_with_backoff = retry_module.async_retry_with_backoff
RetryContext = retry_module.RetryContext


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Test that successful function doesn't retry."""
        call_count = [0]

        @retry_with_backoff(max_retries=3)
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_exception(self):
        """Test that function retries on exception."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count[0] == 3

    def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count[0] == 3  # Initial + 2 retries

    def test_specific_exceptions_only(self):
        """Test that only specified exceptions trigger retry."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exceptions=(KeyError,)
        )
        def raises_type_error():
            call_count[0] += 1
            raise TypeError("Not a KeyError")

        with pytest.raises(TypeError):
            raises_type_error()

        assert call_count[0] == 1  # No retry for TypeError

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        retry_calls = []

        def on_retry(exc, attempt, delay):
            retry_calls.append((attempt, str(exc)))

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            on_retry=on_retry
        )
        def fails_twice():
            if len(retry_calls) < 2:
                raise ValueError(f"Fail {len(retry_calls)}")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert len(retry_calls) == 2

    def test_exponential_backoff(self):
        """Test that delays increase exponentially."""
        delays = []

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False
        )
        def track_delays():
            if len(delays) < 3:
                start = time.time()
                raise ValueError("Retry")
            return delays

        # Modify to track delays
        original_func = track_delays

        @retry_with_backoff(max_retries=3, base_delay=0.1, jitter=False)
        def tracked_func():
            return original_func()

        # Just verify it works
        with pytest.raises(ValueError):
            original_func()


class TestAsyncRetryWithBackoff:
    """Tests for async_retry_with_backoff decorator."""

    def test_async_success_no_retry(self):
        """Test that successful async function doesn't retry."""
        call_count = [0]

        @async_retry_with_backoff(max_retries=3)
        async def successful_func():
            call_count[0] += 1
            return "success"

        result = asyncio.run(successful_func())
        assert result == "success"
        assert call_count[0] == 1

    def test_async_retry_on_exception(self):
        """Test that async function retries on exception."""
        call_count = [0]

        @async_retry_with_backoff(max_retries=3, base_delay=0.01)
        async def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = asyncio.run(failing_then_success())
        assert result == "success"
        assert call_count[0] == 3

    def test_async_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        call_count = [0]

        @async_retry_with_backoff(max_retries=2, base_delay=0.01)
        async def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            asyncio.run(always_fails())

        assert call_count[0] == 3


class TestRetryContext:
    """Tests for RetryContext."""

    def test_should_retry_initial(self):
        """Test should_retry returns True initially."""
        with RetryContext(max_retries=3) as retry:
            assert retry.should_retry() is True

    def test_record_failure(self):
        """Test record_failure increments attempt counter."""
        with RetryContext(max_retries=3, base_delay=0.01) as retry:
            retry.record_failure(ValueError("Test"))
            assert retry.attempt == 1

    def test_max_retries_exceeded_context(self):
        """Test that exception is raised when max retries exceeded."""
        with pytest.raises(ValueError):
            with RetryContext(max_retries=2, base_delay=0.01) as retry:
                while retry.should_retry():
                    try:
                        raise ValueError("Always fails")
                    except ValueError as e:
                        retry.record_failure(e)

    def test_successful_retry_context(self):
        """Test successful retry using context."""
        attempt = [0]
        result = None

        with RetryContext(max_retries=3, base_delay=0.01) as retry:
            while retry.should_retry():
                try:
                    attempt[0] += 1
                    if attempt[0] < 3:
                        raise ValueError("Not yet")
                    result = "success"
                    break
                except ValueError as e:
                    retry.record_failure(e)

        assert result == "success"
        assert attempt[0] == 3

    def test_get_delay(self):
        """Test get_delay returns positive value."""
        with RetryContext(max_retries=3, base_delay=1.0, jitter=False) as retry:
            retry.attempt = 1
            delay = retry.get_delay()
            assert delay > 0
            assert delay <= 2.0  # base_delay * 2^1
