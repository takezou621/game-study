"""Audio configuration dataclass."""

from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio capture configuration."""

    sample_rate: int = 16000  # 16kHz for speech recognition
    channels: int = 1  # Mono
    chunk_size: int = 512  # Frames per chunk
    format: str = "int16"  # Audio format
    device_index: int | None = None  # None for default device
    noise_gate_threshold: float = 0.01  # Noise gate (0-1)
    noise_gate_attack_ms: float = 5.0  # Attack time in ms
    noise_gate_release_ms: float = 50.0  # Release time in ms
    vad_enabled: bool = True  # Enable VAD
    vad_padding_ms: int = 300  # Padding before/after speech
    vad_min_speech_ms: int = 500  # Minimum speech duration
    vad_max_speech_ms: int = 10000  # Maximum speech duration
