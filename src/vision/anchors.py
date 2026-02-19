"""UI anchor detection for calibration."""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Anchor:
    """Represents a detected UI anchor."""
    name: str
    position: Tuple[int, int]  # (x, y) in pixels
    confidence: float


class AnchorDetector:
    """
    UI anchor detector for HUD calibration.

    MVP: Simplified implementation (placeholder for Phase 3).
    Phase 3: Full implementation with YOLO-based anchor detection.
    """

    def __init__(self):
        """Initialize anchor detector."""
        self.enabled = False  # MVP: Disabled

    def detect_anchors(self, frame: np.ndarray) -> Dict[str, Anchor]:
        """
        Detect UI anchor points in frame.

        MVP: Returns default anchor positions based on frame resolution.
        Phase 3: Uses YOLO to detect actual anchor positions.

        Args:
            frame: Input frame

        Returns:
            Dictionary of anchor names to Anchor objects
        """
        if not self.enabled:
            return self._get_default_anchors(frame.shape)

        # Phase 3: Implement actual anchor detection here
        # This is a placeholder for future implementation
        return self._get_default_anchors(frame.shape)

    def _get_default_anchors(self, shape: Tuple[int, ...]) -> Dict[str, Anchor]:
        """
        Get default anchor positions based on frame resolution.

        Args:
            shape: Frame shape (height, width, ...)

        Returns:
            Dictionary of anchor positions
        """
        height, width = shape[:2]

        return {
            "hp_shield_bottom_left": Anchor(
                name="hp_shield_bottom_left",
                position=(int(0.10 * width), int(0.85 * height)),
                confidence=1.0
            ),
            "minimap_top_right": Anchor(
                name="minimap_top_right",
                position=(int(0.90 * width), int(0.10 * height)),
                confidence=1.0
            ),
            "inventory_bottom_right": Anchor(
                name="inventory_bottom_right",
                position=(int(0.90 * width), int(0.85 * height)),
                confidence=1.0
            ),
            "compass_top_center": Anchor(
                name="compass_top_center",
                position=(int(0.50 * width), int(0.05 * height)),
                confidence=1.0
            ),
        }

    def calibrate_roi(
        self,
        detected_anchors: Dict[str, Anchor],
        roi_config: Dict
    ) -> Dict:
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
