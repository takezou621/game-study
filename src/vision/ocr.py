"""OCR detector for extracting numerical values from HUD."""


import cv2
import numpy as np


class OCRDetector:
    """
    OCR detector for extracting numbers and text from HUD.

    MVP: Template-based digit recognition.
    Phase 2+: Integration with Tesseract or specialized OCR models.
    """

    def __init__(self, use_template_matching: bool = True):
        """
        Initialize OCR detector.

        Args:
            use_template_matching: Use template matching for digits (MVP)
        """
        self.use_template_matching = use_template_matching
        self.digit_templates = {}

        if self.use_template_matching:
            self._init_digit_templates()

    def _init_digit_templates(self) -> None:
        """Initialize digit templates for template matching."""
        # MVP: Digit templates would be loaded from training data
        # For now, we'll use basic OCR logic
        pass

    def extract_number(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.7
    ) -> dict:
        """
        Extract number from frame region.

        MVP: Uses template matching + color-based digit recognition.
        Phase 2+: Tesseract/CRNN for more robust OCR.

        Args:
            frame: Input frame (should contain numeric text)
            min_confidence: Minimum confidence threshold

        Returns:
            Detection result with value, confidence, source
        """
        if self.use_template_matching:
            return self._extract_number_template(frame, min_confidence)
        else:
            return self._extract_number_tesseract(frame, min_confidence)

    def _extract_number_template(
        self,
        frame: np.ndarray,
        min_confidence: float
    ) -> dict:
        """
        Extract number using template matching (MVP).

        Args:
            frame: Input frame
            min_confidence: Minimum confidence threshold

        Returns:
            Detection result
        """
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract digits from contours
        digits = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio and size (heuristics for Fortnite HUD digits)
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 0.8 and 10 < h < 50:
                digit_roi = gray[y:y+h, x:x+w]
                digits.append((x, digit_roi))

        # Sort by x position and recognize
        digits.sort(key=lambda x: x[0])

        # MVP: Simple heuristic-based digit recognition
        # In production, use proper digit templates or Tesseract
        recognized_number = 0
        confidence = 0.5

        for _, digit_roi in digits:
            # Resize to standard size for recognition
            resized = cv2.resize(digit_roi, (20, 30), interpolation=cv2.INTER_AREA)
            digit = self._recognize_digit(resized)
            recognized_number = recognized_number * 10 + digit

        return {
            "value": recognized_number,
            "confidence": confidence,
            "source": "ocr_template",
        }

    def _recognize_digit(self, digit_roi: np.ndarray) -> int:
        """
        Recognize single digit (MVP: simplified).

        Args:
            digit_roi: Digit image

        Returns:
            Recognized digit (0-9)
        """
        # MVP: Simple pixel count heuristic
        # In production, use proper template matching or ML
        white_pixels = np.sum(digit_roi > 128)

        # This is a very rough heuristic - real implementation needs templates
        if white_pixels < 50:
            return 1
        elif white_pixels < 100:
            return 7
        elif white_pixels < 150:
            return 4
        elif white_pixels < 200:
            return 3
        elif white_pixels < 250:
            return 5
        else:
            return 0

    def _extract_number_tesseract(
        self,
        frame: np.ndarray,
        min_confidence: float
    ) -> dict:
        """
        Extract number using Tesseract (Phase 2+).

        Args:
            frame: Input frame
            min_confidence: Minimum confidence threshold

        Returns:
            Detection result
        """
        # Phase 2+: Implement Tesseract integration
        # import pytesseract
        # text = pytesseract.image_to_string(frame, config='--psm 7 digits')
        # ...

        return {
            "value": 0,
            "confidence": 0.0,
            "source": "ocr_tesseract",
        }

    def extract_hp(self, frame: np.ndarray) -> dict:
        """
        Extract HP value from frame.

        Args:
            frame: Input frame (HP region)

        Returns:
            HP detection result
        """
        result = self.extract_number(frame)
        result["value"] = min(100, max(0, result["value"]))  # Clamp to 0-100
        return result

    def extract_shield(self, frame: np.ndarray) -> dict:
        """
        Extract Shield value from frame.

        Args:
            frame: Input frame (Shield region)

        Returns:
            Shield detection result
        """
        result = self.extract_number(frame)
        result["value"] = min(100, max(0, result["value"]))  # Clamp to 0-100
        return result

    def extract_ammo(self, frame: np.ndarray) -> dict:
        """
        Extract ammo count from frame.

        Args:
            frame: Input frame (ammo region)

        Returns:
            Ammo detection result
        """
        result = self.extract_number(frame)
        result["value"] = min(999, max(0, result["value"]))  # Clamp to 0-999
        return result
