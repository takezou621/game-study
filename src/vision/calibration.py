"""HUD calibration module for automatic ROI adjustment."""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from .anchors import AnchorDetector, Anchor


@dataclass
class CalibrationResult:
    """Result of HUD calibration."""
    success: bool
    calibrated_roi_config: Dict
    detected_anchors: Dict[str, Dict]
    confidence: float
    frame_size: Tuple[int, int]
    timestamp_ms: float = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.timestamp_ms is None:
            self.timestamp_ms = int(time.time() * 1000)


@dataclass
class CalibrationStats:
    """Statistics for calibration quality."""
    total_detections: int = 0
    successful_calibrations: int = 0
    average_confidence: float = 0.0
    calibration_history: List[Dict] = field(default_factory=list)

    def add_result(self, result: CalibrationResult) -> None:
        """Add calibration result to statistics."""
        self.total_detections += 1
        if result.success:
            self.successful_calibrations += 1

        # Calculate average confidence
        if result.detected_anchors:
            confidences = [a['confidence'] for a in result.detected_anchors.values()]
            avg = sum(confidences) / len(confidences)
            self.average_confidence = (
                self.average_confidence * (self.total_detections - 1) + avg
            ) / self.total_detections

        # Add to history
        self.calibration_history.append({
            'timestamp_ms': result.timestamp_ms,
            'success': result.success,
            'anchor_count': len(result.detected_anchors),
            'confidence': result.confidence
        })

        # Keep only last 100 entries
        if len(self.calibration_history) > 100:
            self.calibration_history.pop(0)

    def get_success_rate(self) -> float:
        """Get calibration success rate."""
        if self.total_detections == 0:
            return 0.0
        return self.successful_calibrations / self.total_detections


class HUDCalibrator:
    """
    Automatic HUD calibration using anchor detection.

    Calibrates ROI positions based on detected UI anchors,
    correcting for HUD scale, safe zone, and resolution differences.
    """

    def __init__(
        self,
        anchor_detector: Optional[AnchorDetector] = None,
        min_anchors_required: int = 2,
        min_calibration_confidence: float = 0.6,
        calibration_window_ms: int = 5000
    ):
        """
        Initialize HUD calibrator.

        Args:
            anchor_detector: Anchor detector instance
            min_anchors_required: Minimum anchors required for successful calibration
            min_calibration_confidence: Minimum overall confidence for calibration
            calibration_window_ms: Time window for considering calibration valid
        """
        self.anchor_detector = anchor_detector or AnchorDetector()
        self.min_anchors_required = min_anchors_required
        self.min_calibration_confidence = min_calibration_confidence
        self.calibration_window_ms = calibration_window_ms

        # Calibration state
        self.current_calibration: Optional[CalibrationResult] = None
        self.stats = CalibrationStats()
        self.last_calibration_time = 0

        # ROI configuration
        self.original_roi_config: Optional[Dict] = None
        self.calibrated_roi_config: Optional[Dict] = None

    def load_roi_config(self, config_path: str) -> None:
        """
        Load ROI configuration from file.

        Args:
            config_path: Path to ROI config file (YAML or JSON)
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"ROI config not found: {config_path}")

        if path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(path, 'r') as f:
                self.original_roi_config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                self.original_roi_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        print(f"Loaded ROI config from: {config_path}")

    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        force: bool = False
    ) -> CalibrationResult:
        """
        Calibrate HUD from a single frame.

        Args:
            frame: Input frame
            force: Force calibration even if recent calibration exists

        Returns:
            CalibrationResult object
        """
        current_time = time.time() * 1000

        # Check if we have a recent valid calibration
        if not force and self._is_calibration_valid(current_time):
            return self.current_calibration

        # Detect anchors
        detected_anchors = self.anchor_detector.detect_anchors(frame)

        if len(detected_anchors) < self.min_anchors_required:
            return CalibrationResult(
                success=False,
                calibrated_roi_config={},
                detected_anchors={},
                confidence=0.0,
                frame_size=frame.shape[:2],
                error_message=f"Insufficient anchors detected: {len(detected_anchors)}/{self.min_anchors_required}"
            )

        # Calculate overall confidence
        confidences = [a.confidence for a in detected_anchors.values()]
        overall_confidence = sum(confidences) / len(confidences)

        if overall_confidence < self.min_calibration_confidence:
            return CalibrationResult(
                success=False,
                calibrated_roi_config={},
                detected_anchors={name: {'position': a.position, 'confidence': a.confidence}
                                 for name, a in detected_anchors.items()},
                confidence=overall_confidence,
                frame_size=frame.shape[:2],
                error_message=f"Confidence too low: {overall_confidence:.2f} < {self.min_calibration_confidence}"
            )

        # Calibrate ROI
        if self.original_roi_config is None:
            return CalibrationResult(
                success=False,
                calibrated_roi_config={},
                detected_anchors={name: {'position': a.position, 'confidence': a.confidence}
                                 for name, a in detected_anchors.items()},
                confidence=overall_confidence,
                frame_size=frame.shape[:2],
                error_message="No ROI config loaded. Call load_roi_config() first."
            )

        calibrated_config = self.anchor_detector.calibrate_roi(
            detected_anchors,
            self.original_roi_config,
            frame.shape[:2]
        )

        # Create result
        result = CalibrationResult(
            success=True,
            calibrated_roi_config=calibrated_config,
            detected_anchors={name: {'position': a.position, 'confidence': a.confidence}
                             for name, a in detected_anchors.items()},
            confidence=overall_confidence,
            frame_size=frame.shape[:2]
        )

        # Update state
        self.current_calibration = result
        self.calibrated_roi_config = calibrated_config
        self.last_calibration_time = current_time
        self.stats.add_result(result)

        return result

    def _is_calibration_valid(self, current_time_ms: float) -> bool:
        """Check if current calibration is still valid."""
        if self.current_calibration is None:
            return False

        age = current_time_ms - self.current_calibration.timestamp_ms
        return age < self.calibration_window_ms

    def get_calibrated_roi(self, roi_name: str) -> Optional[Dict]:
        """
        Get calibrated ROI definition.

        Args:
            roi_name: Name of ROI

        Returns:
            Calibrated ROI definition or None
        """
        if self.calibrated_roi_config is None:
            return None

        return self.calibrated_roi_config.get('rois', {}).get(roi_name)

    def get_all_calibrated_rois(self) -> Dict[str, Dict]:
        """
        Get all calibrated ROI definitions.

        Returns:
            Dictionary of ROI names to calibrated definitions
        """
        if self.calibrated_roi_config is None:
            return {}

        return self.calibrated_roi_config.get('rois', {})

    def save_calibration(self, output_path: str) -> None:
        """
        Save current calibration to file.

        Args:
            output_path: Path to save calibration (YAML or JSON)
        """
        if self.calibrated_roi_config is None:
            raise ValueError("No calibration available to save")

        path = Path(output_path)
        if path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(self.calibrated_roi_config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.calibrated_roi_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")

        print(f"Calibration saved to: {output_path}")

    def visualize_calibration(
        self,
        frame: np.ndarray,
        show_anchors: bool = True,
        show_rois: bool = True
    ) -> np.ndarray:
        """
        Visualize calibration on frame.

        Args:
            frame: Input frame
            show_anchors: Show detected anchor points
            show_rois: Show calibrated ROIs

        Returns:
            Frame with visualization overlay
        """
        viz = frame.copy()

        height, width = frame.shape[:2]

        # Draw anchors
        if show_anchors and self.current_calibration:
            for name, anchor_info in self.current_calibration.detected_anchors.items():
                x, y = anchor_info['position']
                confidence = anchor_info['confidence']

                # Draw anchor point
                color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
                cv2.circle(viz, (x, y), 10, color, -1)
                cv2.circle(viz, (x, y), 15, (255, 255, 255), 2)

                # Draw label
                label = f"{name}: {confidence:.2f}"
                cv2.putText(viz, label, (x + 20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw ROIs
        if show_rois and self.calibrated_roi_config:
            for roi_name, roi_def in self.calibrated_roi_config.get('rois', {}).items():
                bbox = roi_def.get('bbox', [0, 0, 1, 1])

                # Convert normalized to pixel
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)

                # Draw ROI rectangle
                color = (255, 0, 0) if roi_def.get('calibrated') else (0, 255, 255)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = roi_name
                cv2.putText(viz, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw calibration status
        status_text = f"Calibrated: {'Yes' if self.current_calibration and self.current_calibration.success else 'No'}"
        cv2.putText(viz, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.current_calibration:
            conf_text = f"Confidence: {self.current_calibration.confidence:.2f}"
            cv2.putText(viz, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            anchors_text = f"Anchors: {len(self.current_calibration.detected_anchors)}"
            cv2.putText(viz, anchors_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return viz

    def get_calibration_stats(self) -> Dict:
        """
        Get calibration statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_detections': self.stats.total_detections,
            'successful_calibrations': self.stats.successful_calibrations,
            'success_rate': self.stats.get_success_rate(),
            'average_confidence': self.stats.average_confidence,
            'has_current_calibration': self.current_calibration is not None and self.current_calibration.success,
            'last_calibration_age_ms': (
                int((time.time() * 1000) - self.last_calibration_time)
                if self.last_calibration_time > 0 else None
            )
        }

    def reset(self) -> None:
        """Reset calibration state."""
        self.current_calibration = None
        self.calibrated_roi_config = None
        self.last_calibration_time = 0
        print("Calibration state reset")


def create_hud_calibrator(
    roi_config_path: Optional[str] = None,
    min_anchors_required: int = 2
) -> HUDCalibrator:
    """
    Create a HUD calibrator with sensible defaults.

    Args:
        roi_config_path: Path to ROI configuration file
        min_anchors_required: Minimum anchors for successful calibration

    Returns:
        Configured HUDCalibrator instance
    """
    calibrator = HUDCalibrator(min_anchors_required=min_anchors_required)

    if roi_config_path:
        calibrator.load_roi_config(roi_config_path)

    return calibrator
