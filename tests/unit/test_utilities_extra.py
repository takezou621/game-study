"""Extra comprehensive tests for utility modules.

This module provides tests for:
- MetricsCollector (src/utils/metrics.py)
- RateLimiter additional tests (src/utils/rate_limiter.py)
- Time utilities (src/utils/time.py)
- HealthChecker (src/health.py)
"""

import importlib.util
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

# Direct module imports to avoid package dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"


# ============================================================================
# Metrics Tests
# ============================================================================

def _load_metrics_module():
    """Load metrics module directly."""
    metrics_path = SRC_PATH / "utils" / "metrics.py"
    spec = importlib.util.spec_from_file_location("metrics", metrics_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics_module = _load_metrics_module()
MetricsCollector = metrics_module.MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init_default(self):
        """Test MetricsCollector initialization with defaults."""
        collector = MetricsCollector()
        assert collector._max_samples == 1000
        assert collector.get_counter("test") == 0

    def test_init_custom_max_samples(self):
        """Test MetricsCollector with custom max_samples."""
        collector = MetricsCollector(max_samples=100)
        assert collector._max_samples == 100

    def test_increment_default(self):
        """Test increment with default value."""
        collector = MetricsCollector()
        collector.increment("requests")
        assert collector.get_counter("requests") == 1

    def test_increment_custom_value(self):
        """Test increment with custom value."""
        collector = MetricsCollector()
        collector.increment("requests", value=5)
        assert collector.get_counter("requests") == 5

    def test_increment_multiple(self):
        """Test multiple increments."""
        collector = MetricsCollector()
        collector.increment("counter")
        collector.increment("counter", value=2)
        collector.increment("counter", value=3)
        assert collector.get_counter("counter") == 6

    def test_decrement_default(self):
        """Test decrement with default value."""
        collector = MetricsCollector()
        collector.increment("counter", value=10)
        collector.decrement("counter")
        assert collector.get_counter("counter") == 9

    def test_decrement_custom_value(self):
        """Test decrement with custom value."""
        collector = MetricsCollector()
        collector.increment("counter", value=10)
        collector.decrement("counter", value=3)
        assert collector.get_counter("counter") == 7

    def test_decrement_negative(self):
        """Test decrement can go negative."""
        collector = MetricsCollector()
        collector.decrement("counter")
        assert collector.get_counter("counter") == -1

    def test_get_counter_nonexistent(self):
        """Test get_counter returns 0 for nonexistent metric."""
        collector = MetricsCollector()
        assert collector.get_counter("nonexistent") == 0

    def test_reset_counter(self):
        """Test reset_counter returns previous value and resets to 0."""
        collector = MetricsCollector()
        collector.increment("test", value=42)
        previous = collector.reset_counter("test")
        assert previous == 42
        assert collector.get_counter("test") == 0

    def test_reset_counter_nonexistent(self):
        """Test reset_counter with nonexistent metric."""
        collector = MetricsCollector()
        previous = collector.reset_counter("nonexistent")
        assert previous == 0
        assert collector.get_counter("nonexistent") == 0

    def test_reset_all_counters(self):
        """Test reset_all_counters resets all counters."""
        collector = MetricsCollector()
        collector.increment("counter1", value=10)
        collector.increment("counter2", value=20)
        collector.increment("counter3", value=30)

        previous = collector.reset_all_counters()

        assert previous == {"counter1": 10, "counter2": 20, "counter3": 30}
        assert collector.get_counter("counter1") == 0
        assert collector.get_counter("counter2") == 0
        assert collector.get_counter("counter3") == 0

    def test_record_latency_single(self):
        """Test recording a single latency sample."""
        collector = MetricsCollector()
        collector.record_latency(0.123)
        samples = collector.get_latency_samples()
        assert samples == [0.123]

    def test_record_latency_multiple(self):
        """Test recording multiple latency samples."""
        collector = MetricsCollector()
        collector.record_latency(0.1)
        collector.record_latency(0.2)
        collector.record_latency(0.3)
        samples = collector.get_latency_samples()
        assert samples == [0.1, 0.2, 0.3]

    def test_latency_samples_max_limit(self):
        """Test that latency samples are trimmed at max_samples."""
        collector = MetricsCollector(max_samples=5)
        for i in range(10):
            collector.record_latency(float(i))

        samples = collector.get_latency_samples()
        assert len(samples) == 5
        # Should keep most recent (5, 6, 7, 8, 9)
        assert samples == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_clear_latency_samples(self):
        """Test clearing latency samples."""
        collector = MetricsCollector()
        collector.record_latency(0.1)
        collector.record_latency(0.2)
        collector.record_latency(0.3)

        previous = collector.clear_latency_samples()
        assert previous == [0.1, 0.2, 0.3]
        assert collector.get_latency_samples() == []

    def test_get_summary_empty(self):
        """Test get_summary with no data."""
        collector = MetricsCollector()
        summary = collector.get_summary()

        assert summary["counters"] == {}
        assert summary["latency_count"] == 0
        assert summary["latency_min"] == 0.0
        assert summary["latency_max"] == 0.0
        assert summary["latency_avg"] == 0.0
        assert summary["latency_p95"] == 0.0

    def test_get_summary_with_counters(self):
        """Test get_summary with counter data."""
        collector = MetricsCollector()
        collector.increment("requests", value=100)
        collector.increment("errors", value=5)

        summary = collector.get_summary()
        assert summary["counters"] == {"requests": 100, "errors": 5}

    def test_get_summary_with_latency(self):
        """Test get_summary with latency data."""
        collector = MetricsCollector()
        collector.record_latency(0.1)
        collector.record_latency(0.2)
        collector.record_latency(0.3)
        collector.record_latency(0.4)
        collector.record_latency(0.5)

        summary = collector.get_summary()
        assert summary["latency_count"] == 5
        assert summary["latency_min"] == 0.1
        assert summary["latency_max"] == 0.5
        assert summary["latency_avg"] == 0.3  # (0.1+0.2+0.3+0.4+0.5)/5

    def test_get_summary_p95(self):
        """Test 95th percentile calculation."""
        collector = MetricsCollector()
        # 20 samples from 0.01 to 0.20
        for i in range(1, 21):
            collector.record_latency(i / 100.0)

        summary = collector.get_summary()
        # 95th percentile of 20 items: index = ceil(20*0.95)-1 = 18
        # 0-indexed: 18 = 19th item = 0.19
        assert summary["latency_p95"] == 0.19

    def test_thread_safety_counters(self):
        """Test counter operations are thread-safe."""
        collector = MetricsCollector()
        num_threads = 10
        increments_per_thread = 100

        def increment_worker():
            for _ in range(increments_per_thread):
                collector.increment("counter")

        threads = [threading.Thread(target=increment_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.get_counter("counter") == num_threads * increments_per_thread

    def test_thread_safety_latency(self):
        """Test latency recording is thread-safe."""
        collector = MetricsCollector()
        num_threads = 10
        samples_per_thread = 50

        def record_worker():
            for i in range(samples_per_thread):
                collector.record_latency(float(i))

        threads = [threading.Thread(target=record_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        samples = collector.get_latency_samples()
        assert len(samples) == num_threads * samples_per_thread




# ============================================================================
# RateLimiter Additional Tests
# ============================================================================

def _load_rate_limiter_module():
    """Load rate_limiter module directly."""
    path = SRC_PATH / "utils" / "rate_limiter.py"
    spec = importlib.util.spec_from_file_location("rate_limiter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rate_limiter_module = _load_rate_limiter_module()
RateLimiter = rate_limiter_module.RateLimiter
AsyncRateLimiter = rate_limiter_module.AsyncRateLimiter


class TestRateLimiterExtra:
    """Additional tests for RateLimiter."""

    def test_allow_call_decreases_tokens(self):
        """Test that allow_call decreases tokens."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        initial_tokens = limiter._tokens

        limiter.allow_call()
        assert limiter._tokens == initial_tokens - 1.0

    def test_allow_call_below_threshold(self):
        """Test allow_call when tokens < 1 but > 0."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)

        # Use up tokens except partial
        for _ in range(9):
            limiter.allow_call()

        # Use last full token
        limiter.allow_call()

        # Next call should fail
        assert limiter.allow_call() is False

    def test_bucket_capacity_burst(self):
        """Test that bucket capacity allows bursting."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0, bucket_capacity=10)

        # Should be able to make 10 calls immediately (burst)
        allowed = sum(1 for _ in range(10) if limiter.allow_call())
        assert allowed == 10

        # 11th call should fail
        assert limiter.allow_call() is False

    def test_token_accumulation(self):
        """Test tokens accumulate over time."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)

        # Use all tokens
        for _ in range(10):
            limiter.allow_call()
        assert limiter.allow_call() is False

        # Wait for refill
        time.sleep(0.2)  # Should give ~2 tokens

        # Should allow at least one call
        assert limiter.allow_call() is True

    def test_wait_for_token_acquires(self):
        """Test wait_for_token acquires token."""
        limiter = RateLimiter(max_calls=2, period_seconds=0.1)

        # Use both tokens
        limiter.allow_call()
        limiter.allow_call()

        # Wait for token should succeed after refill
        start = time.time()
        result = limiter.wait_for_token(timeout=1.0)
        elapsed = time.time() - start

        assert result is True
        assert elapsed >= 0.05  # Should wait some time for refill

    def test_get_wait_time_increases_when_empty(self):
        """Test get_wait_time increases as tokens deplete."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)

        # Initially should be 0
        assert limiter.get_wait_time() == 0.0

        # Use all tokens
        for _ in range(10):
            limiter.allow_call()

        # Now should have wait time
        wait_time = limiter.get_wait_time()
        assert wait_time > 0
        assert wait_time <= 1.0

    def test_refill_updates_timestamp(self):
        """Test that refill updates the last_update timestamp."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)
        old_timestamp = limiter._last_update

        time.sleep(0.1)
        limiter._refill()

        assert limiter._last_update > old_timestamp

    def test_multiple_refills_consistent(self):
        """Test that multiple refills are consistent."""
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)

        # Use all tokens
        for _ in range(10):
            limiter.allow_call()

        # Wait and check refill
        time.sleep(0.5)  # Should refill ~5 tokens
        limiter._refill()

        # Should have ~5 tokens now
        assert 4 <= limiter._tokens <= 6

    def test_reset_restores_full_capacity(self):
        """Test reset restores tokens to bucket capacity."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0, bucket_capacity=15)

        # Use some tokens
        for _ in range(5):
            limiter.allow_call()

        # Reset
        limiter.reset()

        # Should have full capacity
        assert limiter._tokens == 15.0


class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter."""

    def test_init(self):
        """Test AsyncRateLimiter initialization."""
        limiter = AsyncRateLimiter(max_calls=10, period_seconds=1.0)
        assert limiter.max_calls == 10
        assert limiter.period_seconds == 1.0
        assert limiter.bucket_capacity == 10

    def test_init_custom_capacity(self):
        """Test AsyncRateLimiter with custom capacity."""
        limiter = AsyncRateLimiter(max_calls=10, period_seconds=1.0, bucket_capacity=20)
        assert limiter.bucket_capacity == 20

    def test_allow_call(self):
        """Test allow_call on AsyncRateLimiter."""
        limiter = AsyncRateLimiter(max_calls=5, period_seconds=1.0)

        for _ in range(5):
            assert limiter.allow_call() is True

        # Should be exhausted
        assert limiter.allow_call() is False

    def test_get_wait_time(self):
        """Test get_wait_time on AsyncRateLimiter."""
        limiter = AsyncRateLimiter(max_calls=5, period_seconds=1.0)
        assert limiter.get_wait_time() == 0.0

        limiter.allow_call()
        limiter.allow_call()
        limiter.allow_call()
        limiter.allow_call()
        limiter.allow_call()

        assert limiter.get_wait_time() > 0

    def test_reset(self):
        """Test reset on AsyncRateLimiter."""
        limiter = AsyncRateLimiter(max_calls=1, period_seconds=1.0)
        limiter.allow_call()
        assert limiter.allow_call() is False

        limiter.reset()
        assert limiter.allow_call() is True


# ============================================================================
# Time Utilities Tests
# ============================================================================

def _load_time_module():
    """Load time module directly."""
    path = SRC_PATH / "utils" / "time.py"
    spec = importlib.util.spec_from_file_location("time_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


time_utils = _load_time_module()


class TestTimeUtilities:
    """Tests for time utility functions."""

    def test_get_timestamp_ms(self):
        """Test get_timestamp_ms returns positive integer."""
        ts = time_utils.get_timestamp_ms()
        assert isinstance(ts, int)
        assert ts > 0

    def test_get_timestamp_ms_increases(self):
        """Test get_timestamp_ms values increase over time."""
        ts1 = time_utils.get_timestamp_ms()
        time.sleep(0.01)
        ts2 = time_utils.get_timestamp_ms()
        assert ts2 > ts1

    def test_format_timestamp_default(self):
        """Test format_timestamp with default format."""
        ts = time_utils.get_timestamp_ms()
        formatted = time_utils.format_timestamp(ts)
        assert isinstance(formatted, str)
        # ISO format should contain 'T'
        assert 'T' in formatted

    def test_format_timestamp_custom(self):
        """Test format_timestamp with custom format."""
        ts = 1704067200000  # 2024-01-01 00:00:00 UTC
        formatted = time_utils.format_timestamp(ts, "%Y-%m-%d %H:%M:%S")
        assert "2024-01-01" in formatted

    def test_format_timestamp_past(self):
        """Test format_timestamp with past timestamp."""
        # Known timestamp: 2023-01-01 00:00:00 UTC = 1672531200000 ms
        ts = 1672531200000
        formatted = time_utils.format_timestamp(ts)
        assert "2023" in formatted

    def test_get_elapsed_ms(self):
        """Test get_elapsed_ms calculates correctly."""
        start = time_utils.get_timestamp_ms()
        time.sleep(0.05)  # 50ms
        elapsed = time_utils.get_elapsed_ms(start)
        assert elapsed >= 40  # Allow some tolerance

    def test_get_elapsed_ms_negative(self):
        """Test get_elapsed_ms with future timestamp returns negative."""
        future = time_utils.get_timestamp_ms() + 10000
        elapsed = time_utils.get_elapsed_ms(future)
        assert elapsed < 0


# ============================================================================
# HealthChecker Tests
# ============================================================================

def _load_health_module():
    """Load health module directly."""
    path = SRC_PATH / "health.py"
    spec = importlib.util.spec_from_file_location("health", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health_module = _load_health_module()
check_health = health_module.check_health
get_component_status = health_module.get_component_status


class TestHealthChecker:
    """Tests for HealthChecker functions."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-1234567890abcdef"})
    def test_check_health_with_api_key(self):
        """Test check_health when API key is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required directories
            for dirname in ["logs", "output"]:
                os.makedirs(os.path.join(tmpdir, dirname), exist_ok=True)

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = check_health()
                assert isinstance(result, dict)
                assert "healthy" in result
                assert "config" in result
                assert "api_key" in result
                assert "directories" in result
                assert "details" in result
            finally:
                os.chdir(original_cwd)

    @patch.dict(os.environ, {}, clear=True)
    def test_check_health_without_api_key(self):
        """Test check_health when API key is not set."""
        # API key is optional, so healthy depends on config and directories
        result = check_health()
        assert result["api_key"] is False

    def test_check_health_missing_directories(self):
        """Test check_health when directories don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = check_health()
                assert result["directories"] is False
            finally:
                os.chdir(original_cwd)

    def test_check_health_with_existing_directories(self):
        """Test check_health when directories exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required directories (matching Docker layout)
            for dirname in ["logs", "output"]:
                os.makedirs(os.path.join(tmpdir, dirname), exist_ok=True)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = check_health()
                assert result["directories"] is True
            finally:
                os.chdir(original_cwd)

    def test_check_health_partial_directories(self):
        """Test check_health with partial directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only some directories
            os.makedirs(os.path.join(tmpdir, "logs"), exist_ok=True)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = check_health()
                assert result["directories"] is False
            finally:
                os.chdir(original_cwd)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "not-an-api-key"})
    def test_check_health_invalid_api_key(self):
        """Test check_health with invalid API key format."""
        result = check_health()
        # Key without sk- prefix and too short should fail format check
        assert result["api_key"] is False

    def test_check_health_details_structure(self):
        """Test that health check details have proper structure."""
        result = check_health()
        details = result["details"]

        assert "config" in details
        assert "api_key" in details
        assert "directories" in details

    def test_get_component_status_config(self):
        """Test get_component_status for 'config'."""
        status = get_component_status("config")
        assert status is not None
        assert "healthy" in status
        assert "details" in status

    def test_get_component_status_api_key(self):
        """Test get_component_status for 'api_key'."""
        status = get_component_status("api_key")
        assert status is not None
        assert "healthy" in status
        assert "details" in status

    def test_get_component_status_directories(self):
        """Test get_component_status for 'directories'."""
        status = get_component_status("directories")
        assert status is not None
        assert "healthy" in status
        assert "details" in status

    def test_get_component_status_invalid(self):
        """Test get_component_status with invalid component."""
        status = get_component_status("invalid_component")
        assert status is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-valid-format-123"})
    def test_api_key_format_validation_valid(self):
        """Test API key format validation with valid key."""
        result = check_health()
        assert result["api_key"] is True
        assert result["details"]["api_key"]["key_prefix"].startswith("sk-test")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "not-an-api-key"})
    def test_api_key_format_validation_invalid_format(self):
        """Test API key format validation with invalid format."""
        result = check_health()
        assert result["api_key"] is False

    def test_health_overall_false_on_any_failure(self):
        """Test overall health is False if any component fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = check_health()
                # Without API key set, healthy should be False
                assert result["healthy"] is False
            finally:
                os.chdir(original_cwd)
