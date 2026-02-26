"""UI anchor detection for calibration."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AnchorDetector:
    """UI anchor detector for HUD calibration.

    Uses YOLO detection for anchor icon recognition with fallback to
    default anchor positions based on frame resolution.
    """

    # Anchor icon class names that YOLO might detect
    ANCHOR_CLASSES = [
        "compass",
        "minimap",
        "health_bar",
        "shield_bar",
        "inventory",
        "ammo_counter",
        "weapon_slot",
    ]

    def __init__(
        self,
        yolo_detector: Any = None,
        enabled: bool = True,
        default_width: int = 1920,
        default_height: int = 1080,
    ):
        """Initialize anchor detector.

        Args:
            yolo_detector: Optional YOLODetector instance for icon detection
            enabled: Whether anchor detection is enabled
            default_width: Default screen width for calibration
            default_height: Default screen height for calibration
        """
        self.yolo_detector = yolo_detector
        self.enabled = enabled
        self.default_width = default_width
        self.default_height = default_height

        # Cache for detected anchors
        self._cached_anchors: dict[str, tuple[int, int]] | None = None
        self._cache_frame_hash: int | None = None

    def detect_anchors(
        self, frame: np.ndarray, use_cache: bool = True
    ) -> dict[str, tuple[int, int]]:
        """Detect UI anchor points in frame.

        Uses YOLO detection when available, falls back to default
        anchor positions based on frame resolution.

        Args:
            frame: Input frame
            use_cache: Whether to use cached anchors for same frame

        Returns:
            Dictionary of anchor names to (x, y) coordinates
        """
        if not self.enabled:
            return self._get_default_anchors(frame.shape)

        # Check cache using sampling-based hash (faster for high-res frames)
        frame_hash = self._compute_frame_hash(frame) if use_cache else None
        if use_cache and self._cache_frame_hash == frame_hash and self._cached_anchors:
            return self._cached_anchors

        # Try YOLO-based detection
        if self.yolo_detector and self.yolo_detector.enabled:
            anchors = self._detect_anchors_yolo(frame)
            if anchors:
                self._cached_anchors = anchors
                self._cache_frame_hash = frame_hash
                return anchors

        # Fallback to default anchors
        return self._get_default_anchors(frame.shape)

    def _detect_anchors_yolo(self, frame: np.ndarray) -> dict[str, tuple[int, int]] | None:
        """Detect anchors using YOLO.

        Args:
            frame: Input frame

        Returns:
            Dictionary of anchor positions or None if detection failed
        """
        try:
            detections = self.yolo_detector.detect_icons(frame)

            if not detections:
                return None

            anchors: dict[str, tuple[int, int]] = {}

            # Map detected objects to anchor points
            for det in detections:
                class_name = det["class_name"].lower()
                center = det["center"]

                # Map class names to anchor names
                if "compass" in class_name:
                    anchors["compass_top_center"] = center
                elif "minimap" in class_name or "map" in class_name:
                    anchors["minimap_top_right"] = center
                elif "health" in class_name or "hp" in class_name:
                    anchors["hp_shield_bottom_left"] = center
                elif "inventory" in class_name:
                    anchors["inventory_bottom_right"] = center

            # Only return if we found meaningful anchors
            if len(anchors) >= 2:
                logger.debug(f"Detected {len(anchors)} anchors via YOLO")
                return anchors

            return None

        except Exception as e:
            logger.error(f"YOLO anchor detection failed: {e}")
            return None

    def _get_default_anchors(self, shape: tuple[int, ...]) -> dict[str, tuple[int, int]]:
        """Get default anchor positions based on frame resolution.

        Args:
            shape: Frame shape (height, width, ...)

        Returns:
            Dictionary of anchor positions
        """
        height, width = shape[:2]

        # Standard HUD layout anchors (Fortnite-style)
        return {
            "hp_shield_bottom_left": (int(0.10 * width), int(0.85 * height)),
            "minimap_top_right": (int(0.90 * width), int(0.10 * height)),
            "inventory_bottom_right": (int(0.90 * width), int(0.85 * height)),
            "compass_top_center": (int(0.50 * width), int(0.05 * height)),
        }

    def calibrate_roi(self, detected_anchors: dict[str, tuple[int, int]], roi_config: dict) -> dict:
        """Calibrate ROI positions based on detected anchors.

        Scales ROI coordinates based on the relationship between
        detected anchors and expected positions.

        Args:
            detected_anchors: Detected anchor positions
            roi_config: Original ROI configuration

        Returns:
            Calibrated ROI configuration
        """
        if not self.enabled:
            return roi_config

        if not detected_anchors:
            return roi_config

        # Calculate scale factors from anchor positions
        scale_x, scale_y = self._calculate_scale_factors(detected_anchors)

        if scale_x is None or scale_y is None:
            return roi_config

        # Apply scaling to ROI config
        calibrated = {}
        for roi_name, roi_data in roi_config.items():
            if isinstance(roi_data, dict) and "bbox" in roi_data:
                # Scale normalized coordinates
                bbox = roi_data["bbox"]
                calibrated_bbox = [
                    min(1.0, bbox[0] * scale_x),
                    min(1.0, bbox[1] * scale_y),
                    min(1.0, bbox[2] * scale_x),
                    min(1.0, bbox[3] * scale_y),
                ]
                calibrated[roi_name] = {
                    **roi_data,
                    "bbox": calibrated_bbox,
                    "calibrated": True,
                }
            else:
                calibrated[roi_name] = roi_data

        logger.debug(f"Calibrated ROIs with scale ({scale_x:.3f}, {scale_y:.3f})")
        return calibrated

    def _calculate_scale_factors(
        self, detected_anchors: dict[str, tuple[int, int]]
    ) -> tuple[float | None, float | None]:
        """Calculate scale factors from detected anchors.

        Args:
            detected_anchors: Detected anchor positions

        Returns:
            Tuple of (scale_x, scale_y) or (None, None) if cannot calculate
        """
        # Get default anchors for reference
        default_anchors = self._get_default_anchors((self.default_height, self.default_width))

        # Calculate scale based on minimap position (most reliable anchor)
        if "minimap_top_right" in detected_anchors and "minimap_top_right" in default_anchors:
            detected = detected_anchors["minimap_top_right"]
            default = default_anchors["minimap_top_right"]

            # Scale is ratio of detected to default
            scale_x = detected[0] / default[0] if default[0] != 0 else None
            scale_y = detected[1] / default[1] if default[1] != 0 else None

            return scale_x, scale_y

        # Fallback: use HP/shield anchor
        if (
            "hp_shield_bottom_left" in detected_anchors
            and "hp_shield_bottom_left" in default_anchors
        ):
            detected = detected_anchors["hp_shield_bottom_left"]
            default = default_anchors["hp_shield_bottom_left"]

            scale_x = detected[0] / default[0] if default[0] != 0 else None
            scale_y = detected[1] / default[1] if default[1] != 0 else None

            return scale_x, scale_y

        return None, None

    def get_resolution_from_anchors(
        self, detected_anchors: dict[str, tuple[int, int]]
    ) -> tuple[int, int] | None:
        """Estimate frame resolution from detected anchors.

        Args:
            detected_anchors: Detected anchor positions

        Returns:
            Tuple of (width, height) or None if cannot estimate
        """
        if not detected_anchors:
            return None

        # Get default anchors for reference
        default_anchors = self._get_default_anchors((self.default_height, self.default_width))

        # Estimate from compass (top center) and minimap (top right)
        if "compass_top_center" in detected_anchors and "compass_top_center" in default_anchors:
            # Compass x position is at 50% of width
            compass_x = detected_anchors["compass_top_center"][0]
            estimated_width = compass_x * 2  # 50% * 2 = 100%

            # Use minimap for height estimation
            if "minimap_top_right" in detected_anchors:
                minimap_y = detected_anchors["minimap_top_right"][1]
                # Minimap y is at 10% of height
                estimated_height = minimap_y * 10 if minimap_y > 0 else self.default_height

                return int(estimated_width), int(estimated_height)

        return None

    def reset_cache(self) -> None:
        """Reset the anchor cache."""
        self._cached_anchors = None
        self._cache_frame_hash = None

    def _compute_frame_hash(self, frame: np.ndarray) -> int:
        """Compute a fast hash of the frame using sampling.

        Uses 1/64 sampling for performance on high-res frames.

        Args:
            frame: Input frame

        Returns:
            Hash value of the sampled frame
        """
        # Sample every 8th pixel for fast hashing (1/64 of original data)
        sample = frame[::8, ::8]
        return hash(sample.tobytes())

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/vision.yaml",
        yolo_detector: Any = None,
    ) -> "AnchorDetector":
        """Create AnchorDetector from configuration file.

        Args:
            config_path: Path to vision configuration file
            yolo_detector: Optional YOLO detector instance

        Returns:
            Configured AnchorDetector instance
        """
        import yaml

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError):
            logger.warning(f"Could not load config from {config_path}, using defaults")
            return cls(yolo_detector=yolo_detector)

        anchor_config = config.get("vision", {}).get("anchors", {})

        return cls(
            yolo_detector=yolo_detector,
            enabled=anchor_config.get("enabled", True),
            default_width=anchor_config.get("default_width", 1920),
            default_height=anchor_config.get("default_height", 1080),
        )
