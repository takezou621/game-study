"""WebRTC utilities for State and ROI streaming."""

import asyncio
import json
import time
from typing import Optional, Dict, Any
import numpy as np


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
        stun_servers: Optional[list] = None,
        port_range: tuple = (10000, 20000)
    ):
        """
        Initialize WebRTC streamer.

        Args:
            stun_servers: List of STUN servers for NAT traversal
            port_range: Port range for UDP connections
        """
        if not AIORTC_AVAILABLE:
            print("Warning: aiortc library not available. Running in mock mode.")
            print("Install with: pip install aiortc")

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
        self.max_state_buffer = 100

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
                print(f"Data channel '{channel_name}' opened")

            @self.data_channel.on("close")
            async def on_close():
                self._connected = False
                print(f"Data channel '{channel_name}' closed")

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
            print(f"Invalid JSON received: {message}")

    async def send_state(self, state: Dict[str, Any]) -> bool:
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
            print(f"Failed to send state: {e}")
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
            print("Video track not initialized")
            return False

        try:
            # Convert frame to bytes
            # In full implementation, encode frame and send through video track
            # For MVP, this is a placeholder
            return True

        except Exception as e:
            print(f"Failed to send video frame: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
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
        print("WebRTC connection closed")


class WebRTCSignalingServer:
    """
    Simple signaling server for WebRTC connections.

    MVP: Placeholder implementation.
    Full implementation would use WebSocket-based signaling.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Initialize signaling server.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients = {}

    async def handle_offer(self, offer: dict) -> dict:
        """
        Handle WebRTC offer.

        Args:
            offer: SDP offer

        Returns:
            SDP answer
        """
        # In full implementation, this would create a peer connection
        # and generate an answer
        print(f"Received offer from client")
        return {"status": "received"}

    async def handle_answer(self, answer: dict):
        """
        Handle WebRTC answer.

        Args:
            answer: SDP answer
        """
        print(f"Received answer from client")

    async def handle_ice_candidate(self, candidate: dict):
        """
        Handle ICE candidate.

        Args:
            candidate: ICE candidate
        """
        print(f"Received ICE candidate from client")

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
        }
