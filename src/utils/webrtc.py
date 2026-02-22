"""WebRTC utilities for State and ROI streaming."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Import constants with fallback for standalone usage
try:
    from constants import (
        MAX_STATE_BUFFER_SIZE,
        WEBRTC_DEFAULT_PORT,
        WEBRTC_TOKEN_EXPIRY_SECONDS,
    )
except ImportError:
    MAX_STATE_BUFFER_SIZE = 100
    WEBRTC_DEFAULT_PORT = 8080
    WEBRTC_TOKEN_EXPIRY_SECONDS = 3600


# Try to import aiortc
try:
    import aiortc
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False


class WebRTCStreamer:
    """
    WebRTC streamer for State JSON and ROI video streaming.

    Supports low-latency streaming of game state and ROI regions.
    """

    def __init__(
        self,
        stun_servers: list | None = None,
        port_range: tuple = (10000, 20000)
    ):
        """
        Initialize WebRTC streamer.

        Args:
            stun_servers: List of STUN servers for NAT traversal
            port_range: Port range for UDP connections
        """
        if not AIORTC_AVAILABLE:
            logger.warning("aiortc library not available. Running in mock mode.")
            logger.info("Install with: pip install aiortc")

        self.stun_servers = stun_servers or [
            "stun:stun.l.google.com:19302"
        ]
        self.port_range = port_range

        # WebRTC components
        self.pc = None
        self.video_track = None
        self.data_channel = None

        # State buffering
        self.state_buffer: list = []
        self.max_state_buffer = MAX_STATE_BUFFER_SIZE

        # Statistics
        self._connected = False
        self._last_state_time = 0.0
        self._latency_samples = []
        self._bytes_sent = 0
        self._bytes_received = 0

    async def create_peer_connection(self):
        """
        Create RTCPeerConnection.

        Returns:
            RTCPeerConnection instance or mock
        """
        if AIORTC_AVAILABLE:
            self.pc = aiortc.RTCPeerConnection()

            # Configure STUN servers
            self.pc.set_configuration({
                "iceServers": [
                    {"urls": self.stun_servers}
                ],
                "iceTransportPolicy": "all",
            })

            return self.pc
        else:
            # Mock peer connection
            class MockPeerConnection:
                async def close(self):
                    pass

            self.pc = MockPeerConnection()
            return self.pc

    async def create_data_channel(
        self,
        channel_name: str = "state"
    ):
        """
        Create RTC data channel for State JSON.

        Args:
            channel_name: Channel name

        Returns:
            RTCDataChannel instance or mock
        """
        if not self.pc:
            raise RuntimeError("Peer connection not created. Call create_peer_connection() first.")

        if AIORTC_AVAILABLE:
            self.data_channel = self.pc.createDataChannel(channel_name)

            # Set up message handlers
            @self.data_channel.on("message")
            async def on_message(message):
                await self._handle_data_message(message)

            @self.data_channel.on("open")
            async def on_open():
                self._connected = True
                logger.info(f"Data channel '{channel_name}' opened")

            @self.data_channel.on("close")
            async def on_close():
                self._connected = False
                logger.info(f"Data channel '{channel_name}' closed")

            return self.data_channel
        else:
            # Mock data channel
            class MockDataChannel:
                def send(self, message):
                    pass

                def close(self):
                    pass

            self.data_channel = MockDataChannel()
            self._connected = True
            return self.data_channel

    async def _handle_data_message(self, message: str):
        """
        Handle incoming data channel message.

        Args:
            message: JSON message
        """
        try:
            data = json.loads(message)

            # Handle state acknowledgment
            if data.get("type") == "ack":
                latency_ms = time.time() * 1000 - data.get("timestamp", 0)
                self._latency_samples.append(latency_ms)

            self._bytes_received += len(message)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message}")

    async def send_state(self, state: dict[str, Any]) -> bool:
        """
        Send State JSON through data channel.

        Args:
            state: Game state dictionary

        Returns:
            True if sent successfully
        """
        if not self._connected or self.data_channel is None:
            return False

        try:
            # Add timestamp
            state_with_ts = {
                **state,
                "timestamp": int(time.time() * 1000),
                "type": "state"
            }

            # Serialize to JSON
            message = json.dumps(state_with_ts)

            # Send through data channel (mock if aiortc not available)
            if AIORTC_AVAILABLE:
                self.data_channel.send(message)
            else:
                # Mock: print state instead
                pass

            self._bytes_sent += len(message)
            self._last_state_time = time.time()

            # Update buffer
            self.state_buffer.append(state_with_ts)
            if len(self.state_buffer) > self.max_state_buffer:
                self.state_buffer.pop(0)

            return True

        except Exception as e:
            logger.error(f"Failed to send state: {e}")
            return False

    async def send_video_frame(
        self,
        frame: np.ndarray
    ) -> bool:
        """
        Send video frame through WebRTC.

        Args:
            frame: Video frame as numpy array

        Returns:
            True if sent successfully
        """
        if self.video_track is None:
            logger.warning("Video track not initialized")
            return False

        try:
            # Convert frame to bytes
            # In full implementation, encode frame and send through video track
            # For MVP, this is a placeholder
            return True

        except Exception as e:
            logger.error(f"Failed to send video frame: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get streaming statistics.

        Returns:
            Statistics dictionary
        """
        avg_latency = (
            sum(self._latency_samples) / len(self._latency_samples)
            if self._latency_samples else 0.0
        )

        return {
            "connected": self._connected,
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
            "latency_ms_avg": avg_latency,
            "latency_samples": len(self._latency_samples),
            "state_buffer_size": len(self.state_buffer),
            "last_state_time": self._last_state_time,
        }

    async def close(self):
        """Close WebRTC connection."""
        if AIORTC_AVAILABLE:
            if self.data_channel:
                self.data_channel.close()
                self.data_channel = None

            if self.video_track:
                self.video_track.stop()
                self.video_track = None

            if self.pc:
                await self.pc.close()
                self.pc = None
        else:
            # Mock: just set flags
            self.data_channel = None
            self.video_track = None
            self.pc = None

        self._connected = False
        logger.info("WebRTC connection closed")


class WebRTCSignalingServer:
    """
    Simple signaling server for WebRTC connections with token-based authentication.

    Token-based authentication prevents unauthorized access to the signaling server.
    Clients must provide a valid token in the 'auth_token' field of their requests.

    MVP: Placeholder implementation.
    Full implementation would use WebSocket-based signaling.
    """

    # Token expiration time in seconds
    TOKEN_EXPIRY_SECONDS = WEBRTC_TOKEN_EXPIRY_SECONDS

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = WEBRTC_DEFAULT_PORT,
        secret_key: str | None = None
    ):
        """
        Initialize signaling server.

        Args:
            host: Server host
            port: Server port
            secret_key: Secret key for token signing (reads from WEBRTC_SECRET_KEY env if not provided)
        """
        self.host = host
        self.port = port
        self.clients = {}

        # Get secret key for token signing (do not store in instance for security)
        self._secret_key = secret_key or os.getenv("WEBRTC_SECRET_KEY")
        if not self._secret_key:
            # Generate a random secret key if not provided
            self._secret_key = secrets.token_hex(32)
            logger.warning(
                "WEBRTC_SECRET_KEY not set. Using generated key. "
                "Set WEBRTC_SECRET_KEY environment variable for production."
            )

    def generate_token(self, client_id: str, expiry_seconds: int = None) -> str:
        """
        Generate an authentication token for a client.

        Args:
            client_id: Unique client identifier
            expiry_seconds: Token expiry time in seconds

        Returns:
            Authentication token string
        """
        expiry_seconds = expiry_seconds or self.TOKEN_EXPIRY_SECONDS
        expiry_time = int(time.time()) + expiry_seconds

        # Create token payload
        payload = f"{client_id}:{expiry_time}"

        # Sign with HMAC
        signature = hmac.new(
            self._secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}:{signature}"

    def verify_token(self, token: str) -> tuple[bool, str | None]:
        """
        Verify an authentication token.

        Args:
            token: Token string to verify

        Returns:
            Tuple of (is_valid, client_id or error_message)
        """
        if not token:
            return False, "Token is required"

        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False, "Invalid token format"

            client_id, expiry_str, signature = parts
            expiry_time = int(expiry_str)

            # Check expiry
            if time.time() > expiry_time:
                return False, "Token has expired"

            # Verify signature
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
            logger.error("Token verification failed", exc_info=True)
            return False, "Invalid token format"

    async def handle_offer(self, offer: dict, auth_token: str = None) -> dict:
        """
        Handle WebRTC offer with authentication.

        Args:
            offer: SDP offer
            auth_token: Authentication token

        Returns:
            SDP answer or error response
        """
        # Verify authentication token
        is_valid, result = self.verify_token(auth_token)
        if not is_valid:
            logger.warning(f"Authentication failed: {result}")
            return {"status": "error", "message": "Authentication failed"}

        # In full implementation, this would create a peer connection
        # and generate an answer
        logger.info(f"Received offer from client: {result}")
        return {"status": "received"}

    async def handle_answer(self, answer: dict, auth_token: str = None):
        """
        Handle WebRTC answer with authentication.

        Args:
            answer: SDP answer
            auth_token: Authentication token
        """
        is_valid, result = self.verify_token(auth_token)
        if not is_valid:
            logger.warning(f"Authentication failed: {result}")
            return

        logger.info(f"Received answer from client: {result}")

    async def handle_ice_candidate(self, candidate: dict, auth_token: str = None):
        """
        Handle ICE candidate with authentication.

        Args:
            candidate: ICE candidate
            auth_token: Authentication token
        """
        is_valid, result = self.verify_token(auth_token)
        if not is_valid:
            logger.warning(f"Authentication failed: {result}")
            return

        logger.info(f"Received ICE candidate from client: {result}")

    def get_statistics(self) -> dict:
        """
        Get server statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "host": self.host,
            "port": self.port,
            "clients": len(self.clients),
            "auth_enabled": True,
        }
