"""ROI (Region of Interest) extraction."""


import numpy as np
import yaml


class ROIExtractor:
    """Extract ROIs from frames based on normalized coordinates."""

    def __init__(self, roi_config_path: str):
        """
        Initialize ROI extractor.

        Args:
            roi_config_path: Path to ROI configuration YAML file
        """
        with open(roi_config_path) as f:
            self.config = yaml.safe_load(f)

        self.rois = self.config.get('rois', {})
        self.calibration = self.config.get('calibration', {})

    def get_roi_by_name(self, name: str) -> dict | None:
        """
        Get ROI configuration by name.

        Args:
            name: ROI name (e.g., "hp_shield", "minimap_storm")

        Returns:
            ROI configuration dictionary or None
        """
        return self.rois.get(name)

    def normalized_to_pixel(
        self,
        bbox: list[float],
        width: int,
        height: int
    ) -> tuple[int, int, int, int]:
        """
        Convert normalized bbox to pixel coordinates.

        Args:
            bbox: Normalized bbox [x_min, y_min, x_max, y_max] (0-1)
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Pixel bbox [x_min, y_min, x_max, y_max]
        """
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        return (x_min, y_min, x_max, y_max)

    def extract_roi(
        self,
        frame: np.ndarray,
        roi_name: str
    ) -> np.ndarray | None:
        """
        Extract ROI region from frame.

        Args:
            frame: Input frame
            roi_name: Name of ROI to extract

        Returns:
            ROI image or None if ROI not found
        """
        roi_config = self.get_roi_by_name(roi_name)
        if not roi_config:
            return None

        bbox = roi_config.get('bbox')
        height, width = frame.shape[:2]

        x_min, y_min, x_max, y_max = self.normalized_to_pixel(bbox, width, height)
        return frame[y_min:y_max, x_min:x_max]

    def extract_all_rois(
        self,
        frame: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Extract all defined ROIs from frame.

        Args:
            frame: Input frame

        Returns:
            Dictionary mapping ROI names to ROI images
        """
        rois = {}
        for roi_name in self.rois.keys():
            roi = self.extract_roi(frame, roi_name)
            if roi is not None:
                rois[roi_name] = roi
        return rois

    def get_field_location(
        self,
        roi_name: str,
        field_name: str,
        width: int,
        height: int
    ) -> tuple[int, int, int, int] | None:
        """
        Get pixel coordinates for a specific field within an ROI.

        Args:
            roi_name: Name of ROI
            field_name: Name of field within ROI
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Pixel bbox or None if field not found
        """
        roi_config = self.get_roi_by_name(roi_name)
        if not roi_config:
            return None

        fields = roi_config.get('fields', [])
        for field in fields:
            if field.get('name') == field_name:
                bbox = field.get('location')
                if bbox:
                    return self.normalized_to_pixel(bbox, width, height)

        return None
