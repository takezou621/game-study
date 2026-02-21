"""Audio diagnostics module for echo detection and audio quality analysis.

This module provides algorithms for detecting echo, audio crosstalk,
and measuring audio quality metrics such as SNR, latency, and bitrate.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.logger import get_logger
from utils.exceptions import GameStudyError

logger = get_logger(__name__)


class AudioIssueType(Enum):
    """Types of audio issues that can be detected."""
    ECHO = "echo"
    CROSSTALK = "crosstalk"
    LOW_SNR = "low_snr"
    CLIPPING = "clipping"
    HIGH_LATENCY = "high_latency"
    NO_SIGNAL = "no_signal"


@dataclass
class AudioIssue:
    """Represents a detected audio issue."""
    issue_type: AudioIssueType
    severity: str  # "info", "warning", "error"
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.issue_type.value,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details
        }


@dataclass
class AudioMetrics:
    """Audio quality metrics."""
    snr_db: Optional[float] = None
    latency_ms: Optional[float] = None
    bitrate_kbps: Optional[float] = None
    rms_level: Optional[float] = None
    peak_level: Optional[float] = None
    clipping_count: int = 0
    frequency_spectrum: Optional[List[Tuple[float, float]]] = None
    echo_detected: bool = False
    echo_delay_ms: Optional[float] = None
    crosstalk_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snr_db": self.snr_db,
            "latency_ms": self.latency_ms,
            "bitrate_kbps": self.bitrate_kbps,
            "rms_level": self.rms_level,
            "peak_level": self.peak_level,
            "clipping_count": self.clipping_count,
            "echo_detected": self.echo_detected,
            "echo_delay_ms": self.echo_delay_ms,
            "crosstalk_detected": self.crosstalk_detected
        }


class AudioDiagnosticsError(GameStudyError):
    """Exception raised for audio diagnostics errors."""

    def __init__(
        self,
        message: str,
        check_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        context = context or {}
        context.update({"check_type": check_type})
        super().__init__(message, context=context, cause=cause)
        self.check_type = check_type


class EchoDetector:
    """Detects echo using cross-correlation analysis.

    Echo detection works by comparing the played audio (reference)
    with the captured audio to detect delayed copies.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        echo_threshold: float = 0.3,
        min_delay_ms: float = 50.0,
        max_delay_ms: float = 500.0
    ):
        """Initialize echo detector.

        Args:
            sample_rate: Audio sample rate in Hz
            echo_threshold: Correlation threshold for echo detection (0-1)
            min_delay_ms: Minimum echo delay to detect in milliseconds
            max_delay_ms: Maximum echo delay to detect in milliseconds
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, echo detection will be limited")

        self.sample_rate = sample_rate
        self.echo_threshold = echo_threshold
        self.min_delay_samples = int((min_delay_ms / 1000) * sample_rate)
        self.max_delay_samples = int((max_delay_ms / 1000) * sample_rate)

    async def detect_echo(
        self,
        reference_audio: bytes,
        captured_audio: bytes
    ) -> Tuple[bool, Optional[float], float]:
        """Detect echo in captured audio compared to reference.

        Args:
            reference_audio: Reference audio that was played (PCM16)
            captured_audio: Captured audio that may contain echo (PCM16)

        Returns:
            Tuple of (echo_detected, delay_ms, correlation_strength)
        """
        if not NUMPY_AVAILABLE:
            logger.warning("Echo detection requires NumPy")
            return False, None, 0.0

        try:
            # Convert PCM16 bytes to numpy arrays
            ref = np.frombuffer(reference_audio, dtype=np.int16).astype(np.float32)
            cap = np.frombuffer(captured_audio, dtype=np.int16).astype(np.float32)

            # Normalize
            if np.max(np.abs(ref)) > 0:
                ref = ref / np.max(np.abs(ref))
            if np.max(np.abs(cap)) > 0:
                cap = cap / np.max(np.abs(cap))

            # Pad to same length
            if len(ref) < len(cap):
                ref = np.pad(ref, (0, len(cap) - len(ref)))
            elif len(cap) < len(ref):
                cap = np.pad(cap, (0, len(ref) - len(cap)))

            # Run detection in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._compute_correlation,
                ref,
                cap
            )

        except Exception as e:
            logger.error(f"Echo detection error: {e}", exc_info=True)
            return False, None, 0.0

    def _compute_correlation(
        self,
        ref: np.ndarray,
        cap: np.ndarray
    ) -> Tuple[bool, Optional[float], float]:
        """Compute cross-correlation to detect echo.

        Args:
            ref: Reference audio signal
            cap: Captured audio signal

        Returns:
            Tuple of (echo_detected, delay_ms, correlation_strength)
        """
        # Compute cross-correlation
        correlation = np.correlate(cap, ref, mode='full')

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_value = np.abs(correlation[peak_idx])

        # Normalize correlation
        max_possible = np.sqrt(np.sum(ref**2) * np.sum(cap**2))
        if max_possible > 0:
            correlation_strength = peak_value / max_possible
        else:
            correlation_strength = 0.0

        # Calculate delay (convert index to delay)
        # correlation length is 2*n - 1, center at n-1
        center = len(ref) - 1
        delay_samples = peak_idx - center

        # Check for echo
        echo_detected = (
            correlation_strength >= self.echo_threshold and
            self.min_delay_samples <= abs(delay_samples) <= self.max_delay_samples
        )

        delay_ms = abs(delay_samples) * 1000 / self.sample_rate if echo_detected else None

        return echo_detected, delay_ms, correlation_strength


class CrosstalkDetector:
    """Detects audio crosstalk using frequency spectrum analysis.

    Crosstalk occurs when audio from one channel leaks into another.
    This is detected by analyzing frequency content overlap.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        crosstalk_threshold: float = 0.2,
        analysis_bands: Optional[List[Tuple[int, int]]] = None
    ):
        """Initialize crosstalk detector.

        Args:
            sample_rate: Audio sample rate in Hz
            crosstalk_threshold: Threshold for crosstalk detection
            analysis_bands: Frequency bands to analyze (Hz tuples)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, crosstalk detection limited")

        self.sample_rate = sample_rate
        self.crosstalk_threshold = crosstalk_threshold

        # Default frequency bands for voice analysis
        if analysis_bands is None:
            analysis_bands = [
                (300, 500),    # Low voice
                (500, 2000),   # Mid voice
                (2000, 4000),  # High voice
                (4000, 8000),  # Harmonics
            ]
        self.analysis_bands = analysis_bands

    async def detect_crosstalk(
        self,
        channel_a: bytes,
        channel_b: bytes
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Detect crosstalk between two audio channels.

        Args:
            channel_a: First audio channel (PCM16)
            channel_b: Second audio channel (PCM16)

        Returns:
            Tuple of (crosstalk_detected, crosstalk_level, band_levels)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Crosstalk detection requires SciPy")
            return False, 0.0, {}

        try:
            # Convert to numpy
            a = np.frombuffer(channel_a, dtype=np.int16).astype(np.float32)
            b = np.frombuffer(channel_b, dtype=np.int16).astype(np.float32)

            # Run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._analyze_frequency_crosstalk,
                a,
                b
            )

        except Exception as e:
            logger.error(f"Crosstalk detection error: {e}", exc_info=True)
            return False, 0.0, {}

    def _analyze_frequency_crosstalk(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Analyze frequency spectrum for crosstalk.

        Args:
            a: First channel signal
            b: Second channel signal

        Returns:
            Tuple of (crosstalk_detected, max_crosstalk, band_crosstalk)
        """
        # Ensure same length
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        # Compute FFT
        n = len(a)
        freq = fftfreq(n, 1/self.sample_rate)[:n//2]
        fft_a = np.abs(fft(a)[:n//2])
        fft_b = np.abs(fft(b)[:n//2])

        # Analyze each frequency band
        band_crosstalk = {}
        max_crosstalk = 0.0

        for low, high in self.analysis_bands:
            # Find frequency indices
            band_mask = (freq >= low) & (freq < high)

            if np.sum(band_mask) == 0:
                continue

            # Calculate power in each band
            power_a = np.sum(fft_a[band_mask]**2)
            power_b = np.sum(fft_b[band_mask]**2)

            # Calculate crosstalk (ratio of weaker to stronger)
            if power_a > 0 and power_b > 0:
                ratio = min(power_a, power_b) / max(power_a, power_b)
            else:
                ratio = 0.0

            band_key = f"{low}-{high}Hz"
            band_crosstalk[band_key] = ratio
            max_crosstalk = max(max_crosstalk, ratio)

        crosstalk_detected = max_crosstalk >= self.crosstalk_threshold

        return crosstalk_detected, max_crosstalk, band_crosstalk


class AudioQualityAnalyzer:
    """Analyzes audio quality metrics including SNR, clipping, and levels."""

    def __init__(
        self,
        sample_rate: int = 24000,
        clipping_threshold: float = 0.95,
        min_rms_threshold: float = 0.01,
        target_rms: float = 0.3
    ):
        """Initialize audio quality analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            clipping_threshold: Threshold for clipping detection (0-1)
            min_rms_threshold: Minimum RMS for valid signal
            target_rms: Target RMS level for good quality
        """
        self.sample_rate = sample_rate
        self.clipping_threshold = clipping_threshold
        self.min_rms_threshold = min_rms_threshold
        self.target_rms = target_rms

    async def analyze_quality(
        self,
        audio_data: bytes,
        reference_data: Optional[bytes] = None
    ) -> AudioMetrics:
        """Analyze audio quality metrics.

        Args:
            audio_data: Audio data to analyze (PCM16)
            reference_data: Optional reference for SNR calculation

        Returns:
            AudioMetrics object with quality measurements
        """
        if not NUMPY_AVAILABLE:
            logger.warning("Audio quality analysis requires NumPy")
            return AudioMetrics()

        try:
            # Convert to numpy
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # Normalize to -1 to 1
            if len(audio) > 0:
                audio = audio / 32768.0

            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                None,
                self._compute_metrics,
                audio,
                reference_data
            )

            return metrics

        except Exception as e:
            logger.error(f"Audio quality analysis error: {e}", exc_info=True)
            return AudioMetrics()

    def _compute_metrics(
        self,
        audio: np.ndarray,
        reference_data: Optional[bytes]
    ) -> AudioMetrics:
        """Compute audio quality metrics.

        Args:
            audio: Normalized audio signal
            reference_data: Optional reference for SNR

        Returns:
            AudioMetrics object
        """
        metrics = AudioMetrics()

        if len(audio) == 0:
            return metrics

        # RMS level
        metrics.rms_level = float(np.sqrt(np.mean(audio**2)))

        # Peak level
        metrics.peak_level = float(np.max(np.abs(audio)))

        # Clipping detection
        clipping_samples = np.abs(audio) > self.clipping_threshold
        metrics.clipping_count = int(np.sum(clipping_samples))

        # Frequency spectrum
        if SCIPY_AVAILABLE:
            n = len(audio)
            freq = fftfreq(n, 1/self.sample_rate)[:n//2]
            magnitude = np.abs(fft(audio)[:n//2])

            # Store spectrum points (limited resolution)
            step = max(1, len(freq) // 100)
            metrics.frequency_spectrum = [
                (float(freq[i]), float(magnitude[i]))
                for i in range(0, len(freq), step)
            ]

        # SNR calculation if reference provided
        if reference_data and NUMPY_AVAILABLE:
            ref = np.frombuffer(reference_data, dtype=np.int16).astype(np.float32)
            ref = ref / 32768.0

            # Simple SNR: signal power vs noise floor
            # Noise is difference between signals
            min_len = min(len(audio), len(ref))
            signal = audio[:min_len]
            noise = signal - ref[:min_len]

            signal_power = np.mean(signal**2)
            noise_power = np.mean(noise**2)

            if noise_power > 1e-10:
                metrics.snr_db = float(10 * np.log10(signal_power / noise_power))
            else:
                metrics.snr_db = float('inf')

        return metrics


class AudioDiagnostics:
    """Main audio diagnostics class combining all detection methods."""

    def __init__(
        self,
        sample_rate: int = 24000,
        echo_threshold: float = 0.3,
        crosstalk_threshold: float = 0.2
    ):
        """Initialize audio diagnostics.

        Args:
            sample_rate: Audio sample rate in Hz
            echo_threshold: Echo detection threshold (0-1)
            crosstalk_threshold: Crosstalk detection threshold (0-1)
        """
        self.sample_rate = sample_rate

        self.echo_detector = EchoDetector(
            sample_rate=sample_rate,
            echo_threshold=echo_threshold
        )

        self.crosstalk_detector = CrosstalkDetector(
            sample_rate=sample_rate,
            crosstalk_threshold=crosstalk_threshold
        )

        self.quality_analyzer = AudioQualityAnalyzer(
            sample_rate=sample_rate
        )

        self._last_check_time = 0.0
        self._issue_history: List[AudioIssue] = []

    async def check_audio_loopback(
        self,
        played_audio: bytes,
        captured_audio: bytes
    ) -> Tuple[AudioMetrics, List[AudioIssue]]:
        """Check for echo in audio loopback.

        Args:
            played_audio: Audio that was played
            captured_audio: Audio captured from microphone

        Returns:
            Tuple of (metrics, issues)
        """
        issues = []

        # Detect echo
        echo_detected, echo_delay, correlation = await self.echo_detector.detect_echo(
            played_audio,
            captured_audio
        )

        # Analyze quality
        metrics = await self.quality_analyzer.analyze_quality(captured_audio)

        if echo_detected:
            metrics.echo_detected = True
            metrics.echo_delay_ms = echo_delay

            severity = "warning" if correlation < 0.7 else "error"
            issues.append(AudioIssue(
                issue_type=AudioIssueType.ECHO,
                severity=severity,
                message=f"Echo detected with {correlation:.2f} correlation at {echo_delay:.1f}ms delay",
                details={
                    "correlation": correlation,
                    "delay_ms": echo_delay
                }
            ))

        # Check clipping
        if metrics.clipping_count > 0:
            issues.append(AudioIssue(
                issue_type=AudioIssueType.CLIPPING,
                severity="warning",
                message=f"Audio clipping detected: {metrics.clipping_count} samples",
                details={"clipping_count": metrics.clipping_count}
            ))

        # Check signal level
        if metrics.rms_level is not None and metrics.rms_level < 0.01:
            issues.append(AudioIssue(
                issue_type=AudioIssueType.NO_SIGNAL,
                severity="warning",
                message=f"Low audio signal: RMS={metrics.rms_level:.4f}",
                details={"rms_level": metrics.rms_level}
            ))

        # Check SNR
        if metrics.snr_db is not None and metrics.snr_db < 10:
            issues.append(AudioIssue(
                issue_type=AudioIssueType.LOW_SNR,
                severity="warning" if metrics.snr_db > 5 else "error",
                message=f"Low SNR detected: {metrics.snr_db:.1f} dB",
                details={"snr_db": metrics.snr_db}
            ))

        # Store issues
        self._issue_history.extend(issues)

        # Keep only recent issues (last 100)
        if len(self._issue_history) > 100:
            self._issue_history = self._issue_history[-100:]

        return metrics, issues

    async def check_crosstalk(
        self,
        channel_a: bytes,
        channel_b: bytes
    ) -> Tuple[bool, Dict[str, float], List[AudioIssue]]:
        """Check for crosstalk between audio channels.

        Args:
            channel_a: First audio channel
            channel_b: Second audio channel

        Returns:
            Tuple of (crosstalk_detected, band_levels, issues)
        """
        issues = []

        detected, level, bands = await self.crosstalk_detector.detect_crosstalk(
            channel_a,
            channel_b
        )

        if detected:
            issues.append(AudioIssue(
                issue_type=AudioIssueType.CROSSTALK,
                severity="warning" if level < 0.5 else "error",
                message=f"Crosstalk detected: {level:.2f}",
                details={"bands": bands, "level": level}
            ))

        self._issue_history.extend(issues)
        return detected, bands, issues

    async def measure_latency(
        self,
        start_time: float,
        end_time: float
    ) -> float:
        """Measure audio processing latency.

        Args:
            start_time: Timestamp when audio was sent
            end_time: Timestamp when audio was received

        Returns:
            Latency in milliseconds
        """
        latency_ms = (end_time - start_time) * 1000

        # Check for high latency
        if latency_ms > 200:
            issue = AudioIssue(
                issue_type=AudioIssueType.HIGH_LATENCY,
                severity="warning" if latency_ms < 500 else "error",
                message=f"High latency detected: {latency_ms:.1f}ms",
                details={"latency_ms": latency_ms}
            )
            self._issue_history.append(issue)

        return latency_ms

    def get_issue_history(self, limit: int = 50) -> List[AudioIssue]:
        """Get recent issue history.

        Args:
            limit: Maximum number of issues to return

        Returns:
            List of recent AudioIssue objects
        """
        return self._issue_history[-limit:]

    def clear_history(self) -> None:
        """Clear issue history."""
        self._issue_history.clear()


async def create_audio_diagnostics(
    sample_rate: int = 24000,
    echo_threshold: float = 0.3,
    crosstalk_threshold: float = 0.2
) -> AudioDiagnostics:
    """Create an AudioDiagnostics instance.

    Args:
        sample_rate: Audio sample rate in Hz
        echo_threshold: Echo detection threshold
        crosstalk_threshold: Crosstalk detection threshold

    Returns:
        Configured AudioDiagnostics instance
    """
    return AudioDiagnostics(
        sample_rate=sample_rate,
        echo_threshold=echo_threshold,
        crosstalk_threshold=crosstalk_threshold
    )
