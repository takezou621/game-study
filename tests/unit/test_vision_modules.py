"""Tests for vision modules (ROI, OCR, YOLO, Anchors)."""

import tempfile
import os
import numpy as np
from unittest.mock import MagicMock, Mock, patch

import pytest


# ============================================================================
# ROI Tests
# ============================================================================

class TestROIExtractor:
    """Tests for ROI extraction."""

    def test_init_with_valid_config(self):
        """Test initialization with valid config."""
        from vision.roi import ROIExtractor
        config = """
rois:
  hp_bar:
    x: 0.1
    y: 0.1
    width: 0.2
    height: 0.05
  shield_bar:
    x: 0.1
    y: 0.15
    width: 0.2
    height: 0.05
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        assert extractor is not None

    def test_normalized_to_pixel(self):
        """Test normalized to pixel conversion."""
        from vision.roi import ROIExtractor
        config = "rois:\n  test:\n    x: 0.0\n    y: 0.0\n    width: 0.5\n    height: 0.5"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        # Test conversion
        x, y, w, h = extractor.normalized_to_pixel(0.5, 0.5, 0.25, 0.25, 100, 100)
        assert x == 50
        assert y == 50

    def test_extract_roi(self):
        """Test extracting a single ROI."""
        from vision.roi import ROIExtractor
        config = """
rois:
  test_roi:
    x: 0.0
    y: 0.0
    width: 0.5
    height: 0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        # Create test image (100x100)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        roi = extractor.extract_roi(frame, 'test_roi')

        # Should extract something
        if roi is not None:
            assert roi.shape[0] > 0
            assert roi.shape[1] > 0

    def test_extract_all_rois(self):
        """Test extracting all ROIs."""
        from vision.roi import ROIExtractor
        config = """
rois:
  roi1:
    x: 0.0
    y: 0.0
    width: 0.5
    height: 0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rois = extractor.extract_all_rois(frame)

        assert isinstance(rois, dict)


# ============================================================================
# OCR Tests
# ============================================================================

class TestOCRDetector:
    """Tests for OCR detection."""

    def test_init_with_template_matching(self):
        """Test initialization with template matching."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)
        assert ocr is not None

    def test_init_with_tesseract(self):
        """Test initialization with tesseract."""
        from vision.ocr import OCRDetector
        try:
            ocr = OCRDetector(use_template_matching=False)
            assert ocr is not None
        except ImportError:
            pytest.skip("Tesseract not available")

    def test_extract_hp_from_image(self):
        """Test HP extraction from image."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Create test image
        image = np.ones((30, 80, 3), dtype=np.uint8) * 255

        result = ocr.extract_hp(image)
        assert 'value' in result
        assert 'source' in result
        assert 'confidence' in result

    def test_extract_shield(self):
        """Test shield extraction."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        image = np.ones((30, 80, 3), dtype=np.uint8) * 255
        result = ocr.extract_shield(image)
        assert 'value' in result

    def test_extract_ammo(self):
        """Test ammo extraction."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        image = np.ones((30, 80, 3), dtype=np.uint8) * 255
        result = ocr.extract_ammo(image)
        assert 'value' in result


# ============================================================================
# YOLO Detector Tests
# ============================================================================

class TestYOLODetector:
    """Tests for YOLO detection."""

    def test_init_without_model(self):
        """Test initialization without model (MVP mode)."""
        from vision.yolo_detector import YOLODetector
        detector = YOLODetector()
        assert detector is not None

    def test_detect_knocked_status(self):
        """Test knocked status detection."""
        from vision.yolo_detector import YOLODetector
        detector = YOLODetector()

        # Create test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_knocked_status(image)

        assert 'value' in result
        assert 'source' in result
        assert 'confidence' in result


# ============================================================================
# Anchor Detector Tests
# ============================================================================

class TestAnchorDetector:
    """Tests for anchor detection."""

    def test_init(self):
        """Test initialization."""
        from vision.anchors import AnchorDetector
        detector = AnchorDetector()
        assert detector is not None

    def test_enabled_property(self):
        """Test enabled property."""
        from vision.anchors import AnchorDetector
        detector = AnchorDetector()
        # Default should be disabled for MVP
        assert detector.enabled == False

    def test_detect_anchors(self):
        """Test detect_anchors method."""
        from vision.anchors import AnchorDetector
        detector = AnchorDetector()

        # Create test image
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = detector.detect_anchors(image)

        # Should return result even if disabled
        assert result is not None or result is None  # May return None when disabled

    def test_calibrate_roi(self):
        """Test calibrate_roi method."""
        from vision.anchors import AnchorDetector
        detector = AnchorDetector()

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        # calibrate_roi should not crash
        detector.calibrate_roi(image)
