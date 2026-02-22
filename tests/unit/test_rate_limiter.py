"""Tests for Rate Limiter."""

import time
from pathlib import Path

# Direct module import
SRC_PATH = Path(__file__).parent.parent.parent / "src"
RATE_LIMITER_PATH = SRC_PATH / "utils" / "rate_limiter.py"

import importlib.util

spec = importlib.util.spec_from_file_location("rate_limiter", RATE_LIMITER_PATH)
rate_limiter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rate_limiter_module)

RateLimiter = rate_limiter_module.RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        assert limiter.max_calls == 10
        assert limiter.period_seconds == 1.0
        assert limiter.bucket_capacity == 10

    def test_init_custom_capacity(self):
        """Test RateLimiter with custom bucket capacity."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0, bucket_capacity=20)
        assert limiter.bucket_capacity == 20

    def test_allow_call_initial(self):
        """Test that calls are allowed initially."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)

        for _ in range(5):
            assert limiter.allow_call() is True

    def test_allow_call_exhausted(self):
        """Test that calls are rejected when exhausted."""
        limiter = RateLimiter(max_calls=3, period_seconds=1.0)

        # Use all tokens
        for _ in range(3):
            limiter.allow_call()

        # Next call should be rejected
        assert limiter.allow_call() is False

    def test_token_refill(self):
        """Test that tokens are refilled over time."""
        limiter = RateLimiter(max_calls=10, period_seconds=0.1)  # 100 tokens/sec

        # Use all tokens
        for _ in range(10):
            limiter.allow_call()

        assert limiter.allow_call() is False

        # Wait for refill (0.05s should give ~5 tokens)
        time.sleep(0.05)

        # Should have tokens now
        assert limiter.allow_call() is True

    def test_reset(self):
        """Test reset functionality."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)

        # Use all tokens
        for _ in range(5):
            limiter.allow_call()

        assert limiter.allow_call() is False

        # Reset
        limiter.reset()

        # Should have tokens again
        assert limiter.allow_call() is True

    def test_get_wait_time_zero(self):
        """Test get_wait_time returns 0 when tokens available."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)
        assert limiter.get_wait_time() == 0.0

    def test_get_wait_time_positive(self):
        """Test get_wait_time returns positive value when exhausted."""
        limiter = RateLimiter(max_calls=1, period_seconds=1.0)

        limiter.allow_call()
        wait_time = limiter.get_wait_time()

        assert wait_time > 0
        assert wait_time <= 1.0

    def test_wait_for_token_immediate(self):
        """Test wait_for_token returns immediately when available."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)

        start = time.time()
        result = limiter.wait_for_token(timeout=1.0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 0.1  # Should be nearly instant

    def test_wait_for_token_timeout(self):
        """Test wait_for_token returns False on timeout."""
        limiter = RateLimiter(max_calls=1, period_seconds=10.0)

        # Use the only token
        limiter.allow_call()

        # Wait with short timeout
        result = limiter.wait_for_token(timeout=0.1)

        assert result is False

    def test_rate_limiting_accuracy(self):
        """Test that rate limiting is approximately accurate."""
        limiter = RateLimiter(max_calls=5, period_seconds=0.5)  # 10 calls/sec

        # Make 10 calls (should be allowed: 5 initial + ~5 refilled)
        allowed = 0
        start_time = time.time()

        while time.time() - start_time < 0.6:
            if limiter.allow_call():
                allowed += 1
            else:
                time.sleep(0.01)

        # Should be around 10-12 calls (5 initial + refill)
        assert 8 <= allowed <= 15


class TestRateLimiterThreadSafety:
    """Tests for RateLimiter thread safety."""

    def test_concurrent_access(self):
        """Test concurrent access doesn't cause issues."""
        import threading

        limiter = RateLimiter(max_calls=100, period_seconds=1.0)
        allowed_count = [0]
        lock = threading.Lock()

        def make_calls():
            for _ in range(50):
                if limiter.allow_call():
                    with lock:
                        allowed_count[0] += 1

        threads = [threading.Thread(target=make_calls) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 100 allowed calls
        assert allowed_count[0] == 100
