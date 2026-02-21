"""Audio input/processing module for speech recognition and voice activity detection.

This module provides comprehensive audio functionality including:
- Audio capture from microphone
- Voice activity detection (VAD)
- Speech-to-text (STT) using OpenAI Whisper API
- Real-time streaming transcription
"""

from audio.capture import (
    AudioCapture,
    AudioConfig,
    AudioFrame,
    AudioCaptureError,
    CaptureState,
    SpeechSegment,
    create_audio_capture,
)

from audio.vad import (
    VoiceActivityDetector,
    VADModel,
    VADResult,
    VADConfig,
    StreamingVAD,
    VADError,
    VADNotAvailableError,
    create_vad,
)

from audio.stt_client import (
    STTClient,
    STTModel,
    STTLanguage,
    STTConfig,
    TranscriptionResult,
    PartialTranscription,
    StreamingSTT,
    STTClientError,
    create_stt_client,
)

__all__ = [
    # Capture
    "AudioCapture",
    "AudioConfig",
    "AudioFrame",
    "AudioCaptureError",
    "CaptureState",
    "SpeechSegment",
    "create_audio_capture",
    # VAD
    "VoiceActivityDetector",
    "VADModel",
    "VADResult",
    "VADConfig",
    "StreamingVAD",
    "VADError",
    "VADNotAvailableError",
    "create_vad",
    # STT
    "STTClient",
    "STTModel",
    "STTLanguage",
    "STTConfig",
    "TranscriptionResult",
    "PartialTranscription",
    "StreamingSTT",
    "STTClientError",
    "create_stt_client",
]
