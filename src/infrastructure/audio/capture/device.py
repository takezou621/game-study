"""Audio device management."""

from dataclasses import dataclass
from typing import Any

from infrastructure.exceptions import DeviceNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AudioDeviceInfo:
    """Information about an audio device."""

    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False


def list_audio_devices() -> list[AudioDeviceInfo]:
    """
    List available audio input devices.

    Returns:
        List of AudioDeviceInfo objects
    """
    devices = []

    # Try sounddevice first
    try:
        import sounddevice as sd

        default_device = sd.default.device[0] if sd.default.device else None

        for i, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                devices.append(
                    AudioDeviceInfo(
                        index=i,
                        name=info.get("name", f"Device {i}"),
                        channels=info.get("max_input_channels", 0),
                        sample_rate=info.get("default_samplerate", 0),
                        is_default=(i == default_device),
                    )
                )
        logger.debug(f"Found {len(devices)} audio input devices via sounddevice")
    except ImportError:
        logger.debug("sounddevice not available, trying pyaudio")
    except Exception as e:
        logger.warning(f"Error listing devices via sounddevice: {e}")

    # Fallback to pyaudio
    if not devices:
        try:
            import pyaudio

            p = pyaudio.PyAudio()
            try:
                default_info = p.get_default_input_device_info()
                default_index = default_info["index"]
            except Exception:
                default_index = None

            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append(
                        AudioDeviceInfo(
                            index=i,
                            name=info.get("name", f"Device {i}"),
                            channels=info.get("maxInputChannels", 0),
                            sample_rate=info.get("defaultSampleRate", 0),
                            is_default=(i == default_index),
                        )
                    )
            p.terminate()
            logger.debug(f"Found {len(devices)} audio input devices via pyaudio")
        except ImportError:
            logger.warning("Neither sounddevice nor pyaudio available for audio device listing")
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")

    return devices


def get_default_input_device() -> AudioDeviceInfo | None:
    """
    Get the default input device.

    Returns:
        AudioDeviceInfo for default device, or None if not found
    """
    devices = list_audio_devices()
    for device in devices:
        if device.is_default:
            logger.debug(f"Default input device: {device.name} (index {device.index})")
            return device

    # Return first device if no default marked
    if devices:
        logger.debug(f"No default marked, using first device: {devices[0].name}")
        return devices[0]

    logger.warning("No audio input devices available")
    return None


def validate_device(device_index: int | None, config: Any) -> int:
    """
    Validate and return a valid device index.

    Args:
        device_index: Requested device index or None for default
        config: Audio configuration

    Returns:
        Valid device index

    Raises:
        DeviceNotFoundError: If device cannot be found
    """
    logger.debug(f"Validating audio device: index={device_index}")

    if device_index is not None:
        devices = list_audio_devices()
        for device in devices:
            if device.index == device_index:
                logger.info(f"Using audio device: {device.name} (index {device.index})")
                return device_index

        # Device not found
        available = [d.index for d in devices]
        logger.error(
            f"Audio device {device_index} not found", extra={"available_devices": available}
        )
        raise DeviceNotFoundError(
            device_type="audio",
            device_id=device_index,
            available_devices=available,
        )

    # Get default device
    default = get_default_input_device()
    if default is None:
        logger.error("No audio input devices available")
        raise DeviceNotFoundError(
            device_type="audio",
            device_id=None,
            available_devices=[],
        )

    logger.info(f"Using default audio device: {default.name} (index {default.index})")
    return default.index
