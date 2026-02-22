"""UI anchor detection for calibration."""


import numpy as np


class AnchorDetector:
    """
    UI anchor detector for HUD calibration.

    MVP: Simplified implementation (placeholder for Phase 3).
    Phase 3: Full implementation with YOLO-based anchor detection.
    """

    def __init__(self):
        """Initialize anchor detector."""
        self.enabled = False  # MVP: Disabled

    def detect_anchors(self, frame: np.ndarray) -> dict[str, tuple[int, int]]:
        """
        Detect UI anchor points in frame.

        MVP: Returns default anchor positions based on frame resolution.
        Phase 3: Uses YOLO to detect actual anchor positions.

        Args:
            frame: Input frame

        Returns:
            Dictionary of anchor names to (x, y) coordinates
        """
        if not self.enabled:
            return self._get_default_anchors(frame.shape)

        # Phase 3: Implement actual anchor detection here
        # This is a placeholder for future implementation
        return self._get_default_anchors(frame.shape)

    def _get_default_anchors(self, shape: tuple[int, ...]) -> dict[str, tuple[int, int]]:
        """
        Get default anchor positions based on frame resolution.

        Args:
            shape: Frame shape (height, width, ...)

        Returns:
            Dictionary of anchor positions
        """
        height, width = shape[:2]

        return {
            "hp_shield_bottom_left": (int(0.10 * width), int(0.85 * height)),
            "minimap_top_right": (int(0.90 * width), int(0.10 * height)),
            "inventory_bottom_right": (int(0.90 * width), int(0.85 * height)),
            "compass_top_center": (int(0.50 * width), int(0.05 * height)),
        }

    def calibrate_roi(
        self,
        detected_anchors: dict[str, tuple[int, int]],
        roi_config: dict
    ) -> dict:
        """
        Calibrate ROI positions based on detected anchors.

        MVP: Returns ROI config as-is.
        Phase 3: Applies calibration offsets.

        Args:
            detected_anchors: Detected anchor positions
            roi_config: Original ROI configuration

        Returns:
            Calibrated ROI configuration
        """
        if not self.enabled:
            return roi_config

        # Phase 3: Implement calibration logic here
        # This is a placeholder for future implementation
        return roi_config
