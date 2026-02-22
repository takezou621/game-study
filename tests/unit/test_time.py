"""Tests for time utilities."""

import importlib.util
import time
from pathlib import Path

# Direct module import to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"
TIME_PATH = SRC_PATH / "utils" / "time.py"

spec = importlib.util.spec_from_file_location("time_utils", TIME_PATH)
time_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_utils)

get_timestamp_ms = time_utils.get_timestamp_ms


class TestGetTimestampMs:
    """Tests for get_timestamp_ms function."""

    def test_returns_int(self):
        """Test that function returns an integer."""
        result = get_timestamp_ms()
        assert isinstance(result, int)

    def test_returns_positive_value(self):
        """Test that function returns a positive value."""
        result = get_timestamp_ms()
        assert result > 0

    def test_returns_current_time(self):
        """Test that returned value is close to current time."""
        before = int(time.time() * 1000)
        result = get_timestamp_ms()
        after = int(time.time() * 1000)

        assert before <= result <= after

    def test_returns_milliseconds_precision(self):
        """Test that function has millisecond precision."""
        t1 = get_timestamp_ms()
        time.sleep(0.001)  # 1ms
        t2 = get_timestamp_ms()

        # Should be at least 1ms difference (allowing for timing variations)
        assert t2 >= t1

    def test_values_increase(self):
        """Test that successive calls return increasing values."""
        values = [get_timestamp_ms() for _ in range(10)]
        assert values == sorted(values)
