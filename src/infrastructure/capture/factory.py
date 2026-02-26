"""Capture factory for creating capture instances."""

from typing import Any

from infrastructure.capture.base import BaseCapture
from infrastructure.capture.screen_capture import ScreenCapture
from infrastructure.capture.video_file import VideoFileCapture
from infrastructure.exceptions import ConfigurationLoadError
from utils.logger import get_logger

logger = get_logger(__name__)


class CaptureFactory:
    """Factory for creating capture instances."""

    @staticmethod
    def create_video_capture(
        video_path: str,
        loop: bool = False,
    ) -> VideoFileCapture:
        """
        Create video file capture.

        Args:
            video_path: Path to video file
            loop: Whether to loop video

        Returns:
            VideoFileCapture instance
        """
        return VideoFileCapture(video_path=video_path, loop=loop)

    @staticmethod
    def create_screen_capture(
        monitor_index: int = 1,
        region: tuple[int, int, int, int] | None = None,
    ) -> ScreenCapture:
        """
        Create screen capture.

        Args:
            monitor_index: Monitor to capture
            region: Optional region (x, y, width, height)

        Returns:
            ScreenCapture instance
        """
        return ScreenCapture(monitor_index=monitor_index, region=region)

    @staticmethod
    def create_from_config(config: dict[str, Any]) -> BaseCapture:
        """
        Create capture from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Capture instance

        Raises:
            ValueError: If capture type is not supported
        """
        capture_type = config.get("type", "video")

        if capture_type == "video":
            return CaptureFactory.create_video_capture(
                video_path=config["video_path"],
                loop=config.get("loop", False),
            )
        elif capture_type == "screen":
            return CaptureFactory.create_screen_capture(
                monitor_index=config.get("monitor_index", 1),
                region=config.get("region"),
            )
        else:
            raise ConfigurationLoadError(
                config_path="<config>",
                reason=f"Unsupported capture type: {capture_type}",
                config_type="capture",
            )

    @staticmethod
    def create_auto(
        video_path: str | None = None,
        use_screen: bool = False,
        **kwargs,
    ) -> BaseCapture:
        """
        Auto-select capture type.

        Args:
            video_path: Optional video path
            use_screen: Whether to use screen capture
            **kwargs: Additional arguments

        Returns:
            Capture instance
        """
        if video_path:
            return CaptureFactory.create_video_capture(
                video_path=video_path,
                loop=kwargs.get("loop", False),
            )
        elif use_screen:
            return CaptureFactory.create_screen_capture(
                monitor_index=kwargs.get("monitor_index", 1),
                region=kwargs.get("region"),
            )
        else:
            # Default to screen capture
            logger.info("No capture specified, defaulting to screen capture")
            return CaptureFactory.create_screen_capture()
