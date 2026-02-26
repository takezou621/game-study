"""YOLO-based icon detector for UI elements."""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO detector for UI icons.

    Uses ultralytics YOLO for object detection with graceful degradation
    when the library is not available.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file (null uses yolov8n.pt)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model: Any = None
        self.enabled = False
        self._yolo_available = False

        # Try to load ultralytics
        try:
            from ultralytics import YOLO  # noqa: F401

            self._yolo_available = True
            self._load_model()
        except ImportError:
            logger.warning(
                "ultralytics not available. YOLO detection disabled. "
                "Install with: pip install ultralytics"
            )
            self.enabled = False

    def _load_model(self) -> None:
        """Load YOLO model from ultralytics."""
        if not self._yolo_available:
            return

        try:
            from ultralytics import YOLO

            # Use provided model path or default to yolov8n
            model_to_load = self.model_path or "yolov8n.pt"

            logger.info(f"Loading YOLO model: {model_to_load}")
            self.model = YOLO(model_to_load)
            self.enabled = True
            logger.info("YOLO model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.enabled = False
            self.model = None

    def detect_icons(
        self,
        frame: np.ndarray,
        roi_region: np.ndarray | None = None,
        classes: list[int] | None = None,
    ) -> list[dict]:
        """Detect UI icons in frame or ROI.

        Args:
            frame: Input frame
            roi_region: Optional ROI region to search in (searches full frame if None)
            classes: Optional list of class IDs to filter

        Returns:
            List of detected icons with metadata:
            - class_name: str
            - confidence: float
            - bbox: [x, y, w, h]
            - center: (cx, cy)
        """
        if not self.enabled or self.model is None:
            return []

        # Use ROI region if provided, otherwise use full frame
        search_image = roi_region if roi_region is not None else frame

        try:
            # Run inference
            results = self.model(
                search_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=classes,
                verbose=False,
            )

            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Calculate center and dimensions
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    # Get class info
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names.get(class_id, f"class_{class_id}")
                    confidence = float(box.conf[0].cpu().numpy())

                    detection = {
                        "class_name": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": [int(x1), int(y1), int(w), int(h)],
                        "center": (int(cx), int(cy)),
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def detect_knocked_status(self, frame: np.ndarray) -> dict:
        """Detect knocked down status.

        Uses color-based detection as a fallback when YOLO doesn't have
        a specific class for knocked status icons.

        Args:
            frame: Input frame (should be knocked/revive ROI)

        Returns:
            Detection result with value, confidence, source
        """
        # First try YOLO detection if enabled
        if self.enabled:
            detections = self.detect_icons(frame)
            # Look for person-related classes that might indicate knocked status
            for det in detections:
                if "person" in det["class_name"].lower():
                    return {
                        "value": True,
                        "confidence": det["confidence"],
                        "source": "yolo_detector",
                    }

        # Fallback to color-based detection
        return self._detect_knocked_color(frame)

    def _detect_knocked_color(self, frame: np.ndarray) -> dict:
        """Detect knocked status using color analysis.

        The knocked icon typically has a distinctive red/orange color.

        Args:
            frame: Input frame

        Returns:
            Detection result
        """
        if frame is None or frame.size == 0:
            return {
                "value": False,
                "confidence": 0.0,
                "source": "color_detector",
            }

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color range (knocked icon is typically red)
        # Red wraps around in HSV, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate percentage of red pixels
        red_percentage = np.sum(red_mask > 0) / red_mask.size

        # Threshold for considering it a knocked icon
        knocked_threshold = 0.05  # 5% red pixels
        has_knocked_icon = red_percentage > knocked_threshold

        # Confidence based on how much red is detected
        confidence = min(1.0, red_percentage / 0.2)  # Max at 20% red

        return {
            "value": has_knocked_icon,
            "confidence": confidence,
            "source": "color_detector",
        }

    def detect_weapon_icon(self, frame: np.ndarray) -> str | None:
        """Detect weapon type from icon.

        Args:
            frame: Input frame (should be weapon ROI)

        Returns:
            Weapon name or None if not detected
        """
        if not self.enabled or self.model is None:
            return None

        try:
            # Run detection
            detections = self.detect_icons(frame)

            # Look for weapon-related classes
            weapon_keywords = [
                "gun",
                "rifle",
                "pistol",
                "shotgun",
                "sniper",
                "weapon",
                "assault",
                "smg",
                "launcher",
            ]

            for det in detections:
                class_name = det["class_name"].lower()
                if any(kw in class_name for kw in weapon_keywords):
                    return str(det["class_name"])

            # If no weapon class found, return highest confidence detection
            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                if best["confidence"] >= self.confidence_threshold:
                    return str(best["class_name"])

            return None

        except Exception as e:
            logger.error(f"Weapon detection failed: {e}")
            return None

    def detect_specific_objects(
        self, frame: np.ndarray, target_classes: list[str], min_confidence: float | None = None
    ) -> list[dict]:
        """Detect specific object classes by name.

        Args:
            frame: Input frame
            target_classes: List of class names to detect
            min_confidence: Override confidence threshold

        Returns:
            List of matching detections
        """
        if not self.enabled:
            return []

        detections = self.detect_icons(frame)
        threshold = min_confidence or self.confidence_threshold

        # Filter by target classes and confidence
        matching = [
            d
            for d in detections
            if d["class_name"].lower() in [c.lower() for c in target_classes]
            and d["confidence"] >= threshold
        ]

        return matching

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model info
        """
        if not self.enabled or self.model is None:
            return {
                "enabled": False,
                "model_path": self.model_path,
                "yolo_available": self._yolo_available,
            }

        try:
            return {
                "enabled": True,
                "model_path": self.model_path or "yolov8n.pt",
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "num_classes": len(self.model.names) if hasattr(self.model, "names") else 0,
            }
        except Exception:
            return {
                "enabled": self.enabled,
                "model_path": self.model_path,
            }
