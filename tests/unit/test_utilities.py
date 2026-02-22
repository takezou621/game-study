"""Tests for utility modules (WebRTC, Rate Limiter, Constants)."""




# ============================================================================
# Rate Limiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for rate limiter."""

    def test_init(self):
        """Test initialization."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        assert limiter is not None

    def test_allow_call(self):
        """Test allow_call method."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)

        # Should allow first few calls
        for _ in range(5):
            assert limiter.allow_call() == True

    def test_rate_limit_exceeded(self):
        """Test rate limit is enforced."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(max_calls=2, period_seconds=1.0)

        limiter.allow_call()
        limiter.allow_call()
        # Third call should be blocked
        result = limiter.allow_call()
        assert result == False

    def test_reset(self):
        """Test reset method."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(max_calls=1, period_seconds=1.0)

        limiter.allow_call()
        limiter.reset()
        # Should allow again after reset
        assert limiter.allow_call() == True

    def test_wait_for_token(self):
        """Test wait_for_token method."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)
        # Should return immediately if tokens available
        result = limiter.wait_for_token(timeout=0.1)
        assert result == True

    def test_get_wait_time(self):
        """Test get_wait_time method."""
        from utils.rate_limiter import RateLimiter
        limiter = RateLimiter(max_calls=1, period_seconds=1.0)
        limiter.allow_call()
        wait_time = limiter.get_wait_time()
        assert wait_time >= 0


# ============================================================================
# WebRTC Tests
# ============================================================================

class TestWebRTCSignalingServer:
    """Tests for WebRTC signaling server."""

    def test_init(self):
        """Test initialization."""
        from utils.webrtc import WebRTCSignalingServer
        server = WebRTCSignalingServer(host="127.0.0.1", port=8080)
        assert server is not None

    def test_default_host_port(self):
        """Test default host and port."""
        from utils.webrtc import WebRTCSignalingServer
        server = WebRTCSignalingServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8080


class TestWebRTCStreamer:
    """Tests for WebRTC streamer."""

    def test_init(self):
        """Test initialization."""
        from utils.webrtc import WebRTCStreamer
        streamer = WebRTCStreamer()
        assert streamer is not None


# ============================================================================
# Constants Tests
# ============================================================================

class TestConstants:
    """Tests for constants module."""

    def test_import_constants(self):
        """Test that constants can be imported."""
        from utils.constants import (
            DEFAULT_COOLDOWN_MS,
            MAX_RESPONSE_LENGTH_CHARS,
            MAX_RESPONSE_LENGTH_MS,
        )
        assert DEFAULT_COOLDOWN_MS > 0
        assert MAX_RESPONSE_LENGTH_MS > 0
        assert MAX_RESPONSE_LENGTH_CHARS > 0

    def test_priority_constants(self):
        """Test priority constants."""
        from utils.constants import (
            PRIORITY_CHATTER,
            PRIORITY_LEARNING,
            PRIORITY_SURVIVAL,
            PRIORITY_TACTICAL,
        )
        assert PRIORITY_SURVIVAL == 0
        assert PRIORITY_TACTICAL == 1
        assert PRIORITY_LEARNING == 2
        assert PRIORITY_CHATTER == 3

    def test_hp_shield_constants(self):
        """Test HP and shield constants."""
        from utils.constants import HP_MAX, HP_MIN, SHIELD_MAX, SHIELD_MIN
        assert HP_MIN == 0
        assert HP_MAX == 100
        assert SHIELD_MIN == 0
        assert SHIELD_MAX == 100

    def test_realtime_constants(self):
        """Test realtime API constants."""
        from utils.constants import (
            REALTIME_PREFIX_PADDING_MS,
            REALTIME_SILENCE_DURATION_MS,
            REALTIME_VAD_THRESHOLD,
        )
        assert 0 <= REALTIME_VAD_THRESHOLD <= 1
        assert REALTIME_SILENCE_DURATION_MS > 0
        assert REALTIME_PREFIX_PADDING_MS > 0


# ============================================================================
# Time Utils Tests
# ============================================================================

class TestTimeUtils:
    """Tests for time utilities."""

    def test_get_timestamp_ms(self):
        """Test get_timestamp_ms."""
        from utils.time import get_timestamp_ms
        ts = get_timestamp_ms()
        assert ts > 0
        assert isinstance(ts, int)

    def test_timestamp_increases(self):
        """Test that timestamp increases."""
        import time

        from utils.time import get_timestamp_ms
        ts1 = get_timestamp_ms()
        time.sleep(0.01)
        ts2 = get_timestamp_ms()
        assert ts2 >= ts1
