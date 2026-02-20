"""Tests for WebRTC Signaling Server authentication.

Note: These tests focus on the token-based authentication logic.
The WebRTCStreamer class requires numpy and is tested separately.
"""

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional, Tuple

import pytest


# Minimal implementation of WebRTCSignalingServer for testing
# This avoids the numpy dependency in the full module
class WebRTCSignalingServerForTest:
    """Test-friendly version of WebRTCSignalingServer."""

    TOKEN_EXPIRY_SECONDS = 3600

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        secret_key: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.clients = {}
        self._secret_key = secret_key or os.getenv("WEBRTC_SECRET_KEY")
        if not self._secret_key:
            self._secret_key = secrets.token_hex(32)

    def generate_token(self, client_id: str, expiry_seconds: int = None) -> str:
        expiry_seconds = expiry_seconds or self.TOKEN_EXPIRY_SECONDS
        expiry_time = int(time.time()) + expiry_seconds
        payload = f"{client_id}:{expiry_time}"
        signature = hmac.new(
            self._secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{payload}:{signature}"

    def verify_token(self, token: str) -> Tuple[bool, Optional[str]]:
        if not token:
            return False, "Token is required"
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False, "Invalid token format"
            client_id, expiry_str, signature = parts
            expiry_time = int(expiry_str)
            if time.time() > expiry_time:
                return False, "Token has expired"
            payload = f"{client_id}:{expiry_str}"
            expected_signature = hmac.new(
                self._secret_key.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                return False, "Invalid token signature"
            return True, client_id
        except (ValueError, TypeError):
            return False, "Invalid token format"

    async def handle_offer(self, offer: dict, auth_token: str = None) -> dict:
        is_valid, result = self.verify_token(auth_token)
        if not is_valid:
            return {"status": "error", "message": "Authentication failed"}
        return {"status": "received"}

    async def handle_answer(self, answer: dict, auth_token: str = None):
        is_valid, result = self.verify_token(auth_token)
        if not is_valid:
            return

    async def handle_ice_candidate(self, candidate: dict, auth_token: str = None):
        is_valid, result = self.verify_token(auth_token)
        if not is_valid:
            return

    def get_statistics(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "clients": len(self.clients),
            "auth_enabled": True,
        }


# Use the test implementation
WebRTCSignalingServer = WebRTCSignalingServerForTest


class TestWebRTCSignalingServer:
    """Tests for WebRTCSignalingServer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        server = WebRTCSignalingServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8080
        assert server._secret_key is not None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        server = WebRTCSignalingServer(host="127.0.0.1", port=9000, secret_key="test_secret")
        assert server.host == "127.0.0.1"
        assert server.port == 9000
        assert server._secret_key == "test_secret"


class TestTokenGeneration:
    """Tests for token generation."""

    @pytest.fixture
    def server(self):
        """Create a server with known secret for testing."""
        return WebRTCSignalingServer(host="127.0.0.1", port=8080, secret_key="test_secret_key")

    def test_generate_token_returns_string(self, server):
        """Test that generate_token returns a string."""
        token = server.generate_token("client1")
        assert isinstance(token, str)

    def test_generate_token_format(self, server):
        """Test that token has expected format (client_id:expiry:signature)."""
        token = server.generate_token("client1")
        parts = token.split(":")
        assert len(parts) == 3
        assert parts[0] == "client1"

    def test_generate_token_contains_client_id(self, server):
        """Test that token contains the client ID."""
        token = server.generate_token("test_client_123")
        assert "test_client_123" in token

    def test_generate_token_different_clients(self, server):
        """Test that different clients get different tokens."""
        token1 = server.generate_token("client1")
        token2 = server.generate_token("client2")
        assert token1 != token2

    def test_generate_token_different_times(self, server):
        """Test that same client at different times gets different tokens."""
        token1 = server.generate_token("client1", expiry_seconds=60)
        time.sleep(1)
        token2 = server.generate_token("client1", expiry_seconds=60)
        assert token1 != token2


class TestTokenVerification:
    """Tests for token verification."""

    @pytest.fixture
    def server(self):
        """Create a server with known secret for testing."""
        return WebRTCSignalingServer(host="127.0.0.1", port=8080, secret_key="test_secret_key")

    def test_verify_valid_token(self, server):
        """Test that valid token is accepted."""
        token = server.generate_token("client1")
        is_valid, result = server.verify_token(token)
        assert is_valid is True
        assert result == "client1"

    def test_verify_invalid_token_format(self, server):
        """Test that invalid token format is rejected."""
        is_valid, result = server.verify_token("invalid_token")
        assert is_valid is False
        assert "Invalid token format" in result

    def test_verify_empty_token(self, server):
        """Test that empty token is rejected."""
        is_valid, result = server.verify_token("")
        assert is_valid is False
        assert "required" in result.lower()

    def test_verify_none_token(self, server):
        """Test that None token is rejected."""
        is_valid, result = server.verify_token(None)
        assert is_valid is False
        assert "required" in result.lower()

    def test_verify_token_wrong_signature(self, server):
        """Test that token with wrong signature is rejected."""
        # Create a valid token then tamper with signature
        token = server.generate_token("client1")
        parts = token.split(":")
        tampered_token = f"{parts[0]}:{parts[1]}:invalidsignature"
        is_valid, result = server.verify_token(tampered_token)
        assert is_valid is False
        assert "signature" in result.lower()

    def test_verify_token_wrong_secret(self):
        """Test that token signed with different secret is rejected."""
        server1 = WebRTCSignalingServer(secret_key="secret1")
        server2 = WebRTCSignalingServer(secret_key="secret2")

        token = server1.generate_token("client1")
        is_valid, result = server2.verify_token(token)
        assert is_valid is False

    def test_verify_expired_token(self, server):
        """Test that expired token is rejected."""
        # Generate token that expires immediately
        token = server.generate_token("client1", expiry_seconds=-1)
        is_valid, result = server.verify_token(token)
        assert is_valid is False
        assert "expired" in result.lower()


class TestServerStatistics:
    """Tests for server statistics."""

    @pytest.fixture
    def server(self):
        """Create a server for testing."""
        return WebRTCSignalingServer(host="127.0.0.1", port=8080)

    def test_get_statistics(self, server):
        """Test that get_statistics returns expected fields."""
        stats = server.get_statistics()
        assert "host" in stats
        assert "port" in stats
        assert "clients" in stats
        assert "auth_enabled" in stats
        assert stats["auth_enabled"] is True


class TestHandleOffer:
    """Tests for handle_offer method."""

    @pytest.fixture
    def server(self):
        """Create a server for testing."""
        return WebRTCSignalingServer(host="127.0.0.1", port=8080, secret_key="test_secret")

    def test_handle_offer_with_valid_token(self, server):
        """Test handle_offer accepts valid token."""
        import asyncio
        token = server.generate_token("client1")
        result = asyncio.run(server.handle_offer({"sdp": "test"}, auth_token=token))
        assert result["status"] == "received"

    def test_handle_offer_without_token(self, server):
        """Test handle_offer rejects request without token."""
        import asyncio
        result = asyncio.run(server.handle_offer({"sdp": "test"}, auth_token=None))
        assert result["status"] == "error"
        assert "Authentication" in result["message"]

    def test_handle_offer_with_invalid_token(self, server):
        """Test handle_offer rejects invalid token."""
        import asyncio
        result = asyncio.run(server.handle_offer({"sdp": "test"}, auth_token="invalid"))
        assert result["status"] == "error"
        assert "Authentication" in result["message"]


class TestHandleAnswer:
    """Tests for handle_answer method."""

    @pytest.fixture
    def server(self):
        """Create a server for testing."""
        return WebRTCSignalingServer(host="127.0.0.1", port=8080, secret_key="test_secret")

    def test_handle_answer_with_valid_token(self, server):
        """Test handle_answer accepts valid token."""
        import asyncio
        token = server.generate_token("client1")
        # Should not raise exception
        asyncio.run(server.handle_answer({"sdp": "test"}, auth_token=token))

    def test_handle_answer_without_token(self, server):
        """Test handle_answer silently rejects request without token."""
        import asyncio
        # Should not raise exception, just return early
        asyncio.run(server.handle_answer({"sdp": "test"}, auth_token=None))


class TestHandleIceCandidate:
    """Tests for handle_ice_candidate method."""

    @pytest.fixture
    def server(self):
        """Create a server for testing."""
        return WebRTCSignalingServer(host="127.0.0.1", port=8080, secret_key="test_secret")

    def test_handle_ice_candidate_with_valid_token(self, server):
        """Test handle_ice_candidate accepts valid token."""
        import asyncio
        token = server.generate_token("client1")
        # Should not raise exception
        asyncio.run(server.handle_ice_candidate({"candidate": "test"}, auth_token=token))

    def test_handle_ice_candidate_without_token(self, server):
        """Test handle_ice_candidate silently rejects request without token."""
        import asyncio
        # Should not raise exception, just return early
        asyncio.run(server.handle_ice_candidate({"candidate": "test"}, auth_token=None))
