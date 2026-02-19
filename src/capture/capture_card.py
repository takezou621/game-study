"""Capture card input module for external capture devices."""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaptureDevice:
    """Represents a video capture device."""
    index: int
    name: str
    device_path: Optional[str]
    supported_resolutions: List[Tuple[int, int]]
    max_fps: Optional[float]
    backend: str


class CaptureCardCapture:
    """
    Capture card input for external devices (Elgato Cam Link, etc.).

    Provides optimized capture for capture card devices with
    automatic device detection, resolution optimization, and low latency.
    """

    # Common capture card names for identification
    CAPTURE_CARD_NAMES = [
        'Cam Link',
        'AVerMedia',
        'Elgato',
        'Razer',
        'Logitech',
        'HD60',
        '4K60',
        'Capture',
    ]

    # Recommended capture card settings
    RECOMMENDED_SETTINGS = {
        'resolution': (1920, 1080),  # 1080p
        'fps': 60,
        'pixel_format': 'MJPG',  # Low latency
        'backend': cv2.CAP_ANY,
    }

    def __init__(
        self,
        device_index: Optional[int] = None,
        device_name: Optional[str] = None,
        resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[int] = None,
        auto_detect: bool = True
    ):
        """
        Initialize capture card capture.

        Args:
            device_index: Device index (0, 1, 2, ...)
            device_name: Device name to search for
            resolution: Capture resolution (width, height)
            fps: Frames per second
            auto_detect: Auto-detect capture card devices
        """
        self.device_index = device_index
        self.device_name = device_name
        self.resolution = resolution or self.RECOMMENDED_SETTINGS['resolution']
        self.fps = fps or self.RECOMMENDED_SETTINGS['fps']
        self.auto_detect = auto_detect

        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata: Dict = {}
        self.detected_devices: List[CaptureDevice] = []

        # Capture card specific optimizations
        self.optimize_for_low_latency = True
        self.use_hardware_acceleration = False

    def open(self) -> None:
        """Open capture device."""
        if self.auto_detect:
            self._detect_capture_cards()
            if not self.detected_devices:
                print("Warning: No capture card devices detected")

        # Determine device index
        if self.device_index is None and self.device_name:
            # Find device by name
            self.device_index = self._find_device_by_name(self.device_name)
            if self.device_index is None:
                raise ValueError(f"Device not found: {self.device_name}")

        # Default to first device if not specified
        if self.device_index is None:
            self.device_index = 0

        # Open capture device
        if self.optimize_for_low_latency:
            backend = self.RECOMMENDED_SETTINGS['backend']
            self.cap = cv2.VideoCapture(self.device_index, backend)
        else:
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open capture device: {self.device_index}")

        # Configure device
        self._configure_device()

        # Get device info
        self._collect_metadata()

        print(f"Capture card opened: {self.device_index}")
        if self.metadata:
            print(f"  Resolution: {self.metadata.get('resolution')}")
            print(f"  FPS: {self.metadata.get('fps')}")

    def _detect_capture_cards(self) -> None:
        """Detect available capture card devices."""
        # Scan common device indices
        max_devices = 10

        for i in range(max_devices):
            try:
                temp_cap = cv2.VideoCapture(i, cv2.CAP_ANY)

                if temp_cap.isOpened():
                    # Try to get device name
                    backend = temp_cap.getBackendName()
                    api = temp_cap.getApiPreference()

                    # Check if it's a capture card
                    is_capture_card = self._is_capture_card(backend, api)

                    if is_capture_card:
                        device = CaptureDevice(
                            index=i,
                            name=f"Capture Card {i}",
                            device_path=f"/dev/video{i}",
                            supported_resolutions=[(1920, 1080), (1280, 720)],
                            max_fps=60.0,
                            backend=backend
                        )
                        self.detected_devices.append(device)
                        print(f"  Found: {device.name}")

                    temp_cap.release()

            except Exception as e:
                pass

    def _is_capture_card(self, backend: str, api: int) -> bool:
        """
        Check if device is a capture card.

        Args:
            backend: OpenCV backend name
            api: API preference

        Returns:
            True if device appears to be a capture card
        """
        # Capture cards typically use DirectShow (Windows) or V4L2 (Linux)
        if 'DSHOW' in backend or 'V4L2' in backend:
            return True

        return False

    def _find_device_by_name(self, name: str) -> Optional[int]:
        """
        Find device index by name.

        Args:
            name: Device name or partial name

        Returns:
            Device index or None
        """
        if not self.detected_devices:
            return None

        for device in self.detected_devices:
            if name.lower() in device.name.lower():
                return device.index

        return None

    def _configure_device(self) -> None:
        """Configure capture device settings."""
        width, height = self.resolution

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Low latency settings
        if self.optimize_for_low_latency:
            # Set buffer size to 1 for minimal latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Disable auto exposure for more consistent results
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

            # Set exposure (if supported)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    def _collect_metadata(self) -> None:
        """Collect device metadata."""
        if not self.cap:
            return

        # Get actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.metadata = {
            'resolution': (actual_width, actual_height),
            'fps': actual_fps,
            'backend': self.cap.getBackendName(),
            'api': self.cap.getApiPreference(),
            'buffer_size': self.cap.get(cv2.CAP_PROP_BUFFERSIZE),
        }

    def read(self) -> Optional[np.ndarray]:
        """
        Read frame from capture card.

        Returns:
            Frame as numpy array or None if failed
        """
        if not self.cap:
            raise RuntimeError("Capture device not opened. Call open() first.")

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def close(self) -> None:
        """Close capture device."""
        if self.cap:
            self.cap.release()
            self.cap = None
            print("Capture card closed")

    def __iter__(self):
        """Iterate over frames."""
        self.open()
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.close()

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def get_metadata(self) -> Dict:
        """
        Get device metadata.

        Returns:
            Dictionary with device information
        """
        return self.metadata.copy()

    def list_capture_cards(self) -> List[CaptureDevice]:
        """
        List available capture card devices.

        Returns:
            List of detected capture card devices
        """
        if not self.detected_devices:
            self._detect_capture_cards()

        return self.detected_devices.copy()

    @staticmethod
    def get_available_devices(max_devices: int = 10) -> List[Dict]:
        """
        Get all available video capture devices.

        Args:
            max_devices: Maximum device index to check

        Returns:
            List of device information dictionaries
        """
        devices = []

        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_ANY)

                if cap.isOpened():
                    device_info = {
                        'index': i,
                        'backend': cap.getBackendName(),
                        'api': cap.getApiPreference(),
                    }

                    # Try to get resolution
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                    if width > 0 and height > 0:
                        device_info['resolution'] = (int(width), int(height))

                    devices.append(device_info)
                    cap.release()

            except Exception as e:
                pass

        return devices


def create_capture_card_capture(
    device_index: Optional[int] = None,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 60
) -> CaptureCardCapture:
    """
    Create a capture card capture with sensible defaults.

    Args:
        device_index: Device index
        resolution: Capture resolution (width, height)
        fps: Frames per second

    Returns:
        Configured CaptureCardCapture instance
    """
    return CaptureCardCapture(
        device_index=device_index,
        resolution=resolution,
        fps=fps,
        auto_detect=True
    )
