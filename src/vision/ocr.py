"""OCR detector for extracting numerical values from HUD."""

from pathlib import Path

import cv2
import numpy as np

from vision.templates import DIGITS_DIR, TEMPLATE_HEIGHT, TEMPLATE_WIDTH


class OCRDetector:
    """OCR detector for extracting numbers and text from HUD.

    Uses template matching with normalized cross-correlation (NCC) for
    robust digit recognition. Falls back to Tesseract if available.
    """

    def __init__(self, use_template_matching: bool = True, templates_dir: Path | None = None):
        """Initialize OCR detector.

        Args:
            use_template_matching: Use template matching for digits
            templates_dir: Custom templates directory (default: built-in)
        """
        self.use_template_matching = use_template_matching
        self.digit_templates: dict[int, np.ndarray] = {}
        self.templates_dir = templates_dir or DIGITS_DIR

        if self.use_template_matching:
            self._init_digit_templates()

        # Try to import pytesseract for fallback
        self._tesseract_available = False
        try:
            import pytesseract  # noqa: F401

            self._tesseract_available = True
        except ImportError:
            pass

    def _init_digit_templates(self) -> None:
        """Initialize digit templates for template matching.

        Loads templates from the templates directory. If templates don't exist,
        generates synthetic templates automatically.
        """
        # Ensure templates directory exists
        if not self.templates_dir.exists():
            self._generate_templates()

        # Load templates for digits 0-9
        for digit in range(10):
            template_path = self.templates_dir / f"{digit}.png"
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.digit_templates[digit] = template
            else:
                # Generate missing template on-the-fly
                self.digit_templates[digit] = self._create_digit_template(digit)

        # Verify we have all templates
        if len(self.digit_templates) != 10:
            raise RuntimeError(
                f"Failed to load digit templates. Only loaded {len(self.digit_templates)}"
            )

    def _generate_templates(self) -> None:
        """Generate synthetic digit templates."""
        from vision.templates.generate_templates import create_synthetic_templates

        create_synthetic_templates(self.templates_dir)

    def _create_digit_template(self, digit: int) -> np.ndarray:
        """Create a single digit template on-the-fly.

        Args:
            digit: The digit (0-9) to create

        Returns:
            Grayscale template array
        """
        # Simple pattern-based digit generation
        template = np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.uint8)

        # Define patterns for each digit (scaled to 20x30)
        patterns = {
            0: [
                (2, 1),
                (3, 1),
                (4, 1),
                (1, 2),
                (5, 2),
                (1, 3),
                (5, 3),
                (1, 4),
                (5, 4),
                (1, 5),
                (5, 5),
                (1, 6),
                (5, 6),
                (2, 7),
                (3, 7),
                (4, 7),
            ],
            1: [(3, 1), (2, 2), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (2, 7), (3, 7), (4, 7)],
            2: [
                (2, 1),
                (3, 1),
                (4, 1),
                (1, 2),
                (5, 2),
                (5, 3),
                (4, 4),
                (3, 5),
                (2, 6),
                (1, 7),
                (2, 7),
                (3, 7),
                (4, 7),
                (5, 7),
            ],
            3: [
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 2),
                (5, 3),
                (4, 4),
                (5, 5),
                (5, 6),
                (1, 7),
                (2, 7),
                (3, 7),
                (4, 7),
            ],
            4: [
                (1, 1),
                (5, 1),
                (1, 2),
                (5, 2),
                (1, 3),
                (5, 3),
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
                (5, 5),
                (5, 6),
                (5, 7),
            ],
            5: [
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 5),
                (5, 6),
                (1, 7),
                (2, 7),
                (3, 7),
                (4, 7),
            ],
            6: [
                (2, 1),
                (3, 1),
                (4, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
                (1, 6),
                (5, 6),
                (2, 7),
                (3, 7),
                (4, 7),
            ],
            7: [
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (5, 2),
                (4, 3),
                (4, 4),
                (3, 5),
                (3, 6),
                (3, 7),
            ],
            8: [
                (2, 1),
                (3, 1),
                (4, 1),
                (1, 2),
                (5, 2),
                (1, 3),
                (5, 3),
                (2, 4),
                (3, 4),
                (4, 4),
                (1, 5),
                (5, 5),
                (1, 6),
                (5, 6),
                (2, 7),
                (3, 7),
                (4, 7),
            ],
            9: [
                (2, 1),
                (3, 1),
                (4, 1),
                (1, 2),
                (5, 2),
                (1, 3),
                (5, 3),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
                (5, 5),
                (5, 6),
                (2, 7),
                (3, 7),
                (4, 7),
            ],
        }

        if digit in patterns:
            scale_x = TEMPLATE_WIDTH / 7
            scale_y = TEMPLATE_HEIGHT / 9
            for gx, gy in patterns[digit]:
                x = int(gx * scale_x)
                y = int(gy * scale_y)
                cv2.rectangle(template, (x, y), (x + 2, y + 3), 255, -1)

        return template

    def extract_number(self, frame: np.ndarray, min_confidence: float = 0.7) -> dict:
        """Extract number from frame region.

        Uses template matching for robust digit recognition with confidence
        scores based on normalized cross-correlation.

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

    def _extract_number_template(self, frame: np.ndarray, _min_confidence: float) -> dict:
        """Extract number using template matching.

        Args:
            frame: Input frame
            _min_confidence: Minimum confidence threshold (unused in template mode)

        Returns:
            Detection result with recognized number and confidence
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Apply adaptive thresholding for binarization
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours for digit segmentation
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract digit regions
        digit_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio and size (heuristics for HUD digits)
            aspect_ratio = w / max(h, 1)
            if 0.15 < aspect_ratio < 1.0 and 8 < h < 60 and 3 < w < 40:
                digit_roi = gray[y : y + h, x : x + w]
                digit_regions.append((x, digit_roi))

        # Sort by x position (left to right)
        digit_regions.sort(key=lambda d: d[0])

        if not digit_regions:
            return {
                "value": 0,
                "confidence": 0.0,
                "source": "ocr_template",
            }

        # Recognize each digit using template matching
        recognized_digits = []
        confidences = []

        for _, digit_roi in digit_regions:
            digit, confidence = self._recognize_digit(digit_roi)
            recognized_digits.append(digit)
            confidences.append(confidence)

        # Combine digits into a number
        if not recognized_digits:
            return {
                "value": 0,
                "confidence": 0.0,
                "source": "ocr_template",
            }

        number = 0
        for digit in recognized_digits:
            number = number * 10 + digit

        # Average confidence across all digits
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "value": number,
            "confidence": avg_confidence,
            "source": "ocr_template",
        }

    def _recognize_digit(self, digit_roi: np.ndarray) -> tuple[int, float]:
        """Recognize single digit using template matching with NCC.

        Args:
            digit_roi: Grayscale digit image region

        Returns:
            Tuple of (recognized digit 0-9, confidence score 0.0-1.0)
        """
        # Resize to standard template size
        try:
            resized = cv2.resize(
                digit_roi, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA
            )
        except cv2.error:
            return 0, 0.0

        # Apply threshold to create binary image
        _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)

        best_match = 0
        best_score = -1.0

        # Compare against all digit templates using normalized cross-correlation
        for digit, template in self.digit_templates.items():
            # Ensure template is same size
            if template.shape != binary.shape:
                template = cv2.resize(
                    template, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA
                )

            # Calculate normalized cross-correlation
            result = cv2.matchTemplate(
                binary.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_match = digit

        # Convert NCC score to confidence (0.0 to 1.0)
        # NCC ranges from -1 to 1, so we map it appropriately
        confidence = max(0.0, (best_score + 1.0) / 2.0)

        return best_match, confidence

    def _extract_number_tesseract(self, frame: np.ndarray, _min_confidence: float) -> dict:
        """Extract number using Tesseract OCR.

        Args:
            frame: Input frame
            _min_confidence: Minimum confidence threshold (unused in Tesseract mode)

        Returns:
            Detection result
        """
        if not self._tesseract_available:
            return {
                "value": 0,
                "confidence": 0.0,
                "source": "ocr_tesseract_unavailable",
            }

        try:
            import pytesseract

            # Preprocess for Tesseract
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Configure Tesseract for digit recognition
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"

            # Run OCR
            data = pytesseract.image_to_data(
                binary, config=config, output_type=pytesseract.Output.DICT
            )

            # Extract text and confidence
            texts = [t for t in data.get("text", []) if t.strip().isdigit()]
            confs = [
                c
                for c, t in zip(data.get("conf", []), data.get("text", []), strict=False)
                if t.strip().isdigit()
            ]

            if texts:
                try:
                    value = int("".join(texts))
                    confidence = sum(confs) / len(confs) / 100.0 if confs else 0.5
                except ValueError:
                    value = 0
                    confidence = 0.0
            else:
                value = 0
                confidence = 0.0

            return {
                "value": value,
                "confidence": confidence,
                "source": "ocr_tesseract",
            }

        except Exception:
            return {
                "value": 0,
                "confidence": 0.0,
                "source": "ocr_tesseract_error",
            }

    def extract_hp(self, frame: np.ndarray) -> dict:
        """Extract HP value from frame.

        Args:
            frame: Input frame (HP region)

        Returns:
            HP detection result (clamped to 0-100)
        """
        result = self.extract_number(frame)
        result["value"] = min(100, max(0, result["value"]))
        return result

    def extract_shield(self, frame: np.ndarray) -> dict:
        """Extract Shield value from frame.

        Args:
            frame: Input frame (Shield region)

        Returns:
            Shield detection result (clamped to 0-100)
        """
        result = self.extract_number(frame)
        result["value"] = min(100, max(0, result["value"]))
        return result

    def extract_ammo(self, frame: np.ndarray) -> dict:
        """Extract ammo count from frame.

        Args:
            frame: Input frame (ammo region)

        Returns:
            Ammo detection result (clamped to 0-999)
        """
        result = self.extract_number(frame)
        result["value"] = min(999, max(0, result["value"]))
        return result
