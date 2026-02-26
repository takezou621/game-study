"""Noise gate audio processing."""

import numpy as np


class NoiseGate:
    """
    Noise gate processor for audio signals.

    Applies a configurable noise gate with attack and release times
    to reduce background noise in audio signals.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize noise gate.

        Args:
            threshold: Gate threshold (0-1)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            sample_rate: Audio sample rate
        """
        self.threshold = threshold
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.sample_rate = sample_rate

        # Calculate coefficients
        self.attack_coeff = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
        self.release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000))

        # State
        self._gate_open = False
        self._envelope = 0.0

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply noise gate to audio data.

        Args:
            audio_data: Input audio samples (float32, normalized -1 to 1)

        Returns:
            Gated audio samples
        """
        result = np.zeros_like(audio_data)

        for i, sample in enumerate(audio_data):
            # Calculate envelope
            envelope = abs(sample)

            # Gate logic
            if self._gate_open:
                if envelope < self.threshold:
                    self._envelope *= self.release_coeff
                    if self._envelope < self.threshold:
                        self._gate_open = False
                else:
                    self._envelope = max(self._envelope, envelope)
            else:
                if envelope > self.threshold:
                    self._envelope = envelope
                    self._gate_open = True
                else:
                    self._envelope *= self.attack_coeff

            # Apply gain (minimum gain when closed)
            gain = self._envelope if self._gate_open else 0.01
            result[i] = sample * gain

        return result

    def reset(self) -> None:
        """Reset gate state."""
        self._gate_open = False
        self._envelope = 0.0

    @property
    def is_open(self) -> bool:
        """Check if gate is currently open."""
        return self._gate_open


class SimpleNoiseGate:
    """Simpler noise gate without envelope following."""

    def __init__(self, threshold: float = 0.01):
        """
        Initialize simple noise gate.

        Args:
            threshold: Gate threshold (0-1)
        """
        self.threshold = threshold

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply simple noise gate.

        Args:
            audio_data: Input audio samples

        Returns:
            Gated audio (silence below threshold)
        """
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < self.threshold:
            return np.zeros_like(audio_data)
        return audio_data
