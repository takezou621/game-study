"""Audio capture infrastructure modules."""

from infrastructure.audio.capture.capture import (
    AudioCapture,
    AudioCaptureError,
    CaptureState,
    DeviceNotFoundError,
)
from infrastructure.audio.capture.config import AudioConfig
from infrastructure.audio.capture.device import (
    AudioDeviceInfo,
    get_default_input_device,
    list_audio_devices,
)
from infrastructure.audio.processing.buffer import AudioBuffer, SpeechBuffer
from infrastructure.audio.processing.noise_gate import NoiseGate

__all__ = [
    # Device management
    "list_audio_devices",
    "get_default_input_device",
    "AudioDeviceInfo",
    # Capture
    "AudioCapture",
    "AudioCaptureError",
    "DeviceNotFoundError",
    "CaptureState",
    # Config
    "AudioConfig",
    # Processing
    "NoiseGate",
    "AudioBuffer",
    "SpeechBuffer",
]
