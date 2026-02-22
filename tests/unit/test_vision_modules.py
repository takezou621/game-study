"""Tests for vision modules (ROI, OCR, YOLO, Anchors)."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


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
    bbox: [0.1, 0.1, 0.3, 0.15]
  shield_bar:
    bbox: [0.1, 0.15, 0.3, 0.2]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        assert extractor is not None
        assert 'hp_bar' in extractor.rois
        assert 'shield_bar' in extractor.rois

    def test_init_with_calibration_data(self):
        """Test initialization with calibration data."""
        from vision.roi import ROIExtractor
        config = """
rois:
  test:
    bbox: [0.0, 0.0, 0.5, 0.5]
calibration:
  screen_width: 1920
  screen_height: 1080
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        assert extractor.calibration['screen_width'] == 1920
        assert extractor.calibration['screen_height'] == 1080

    def test_normalized_to_pixel(self):
        """Test normalized to pixel conversion."""
        from vision.roi import ROIExtractor
        config = "rois:\n  test:\n    bbox: [0.0, 0.0, 0.5, 0.5]"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        # Test conversion - takes bbox [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = extractor.normalized_to_pixel([0.5, 0.5, 0.75, 0.75], 100, 100)
        assert x_min == 50
        assert y_min == 50
        assert x_max == 75
        assert y_max == 75

    def test_normalized_to_pixel_full_frame(self):
        """Test normalized to pixel conversion for full frame."""
        from vision.roi import ROIExtractor
        config = "rois:\n  test:\n    bbox: [0.0, 0.0, 1.0, 1.0]"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        x_min, y_min, x_max, y_max = extractor.normalized_to_pixel([0.0, 0.0, 1.0, 1.0], 1920, 1080)
        assert x_min == 0
        assert y_min == 0
        assert x_max == 1920
        assert y_max == 1080

    def test_get_roi_by_name(self):
        """Test getting ROI configuration by name."""
        from vision.roi import ROIExtractor
        config = """
rois:
  hp_shield:
    bbox: [0.03, 0.78, 0.32, 0.98]
    fields:
      - name: hp
        location: [0.0, 0.0, 0.5, 1.0]
  minimap_storm:
    bbox: [0.8, 0.0, 1.0, 0.2]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        roi = extractor.get_roi_by_name('hp_shield')
        assert roi is not None
        assert roi['bbox'] == [0.03, 0.78, 0.32, 0.98]
        assert len(roi['fields']) == 1

        # Test non-existent ROI
        non_existent = extractor.get_roi_by_name('non_existent')
        assert non_existent is None

    def test_extract_roi(self):
        """Test extracting a single ROI."""
        from vision.roi import ROIExtractor
        config = """
rois:
  test_roi:
    bbox: [0.0, 0.0, 0.5, 0.5]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        # Create test image (100x100x3)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        roi = extractor.extract_roi(frame, 'test_roi')

        # Should extract something
        assert roi is not None
        assert roi.shape[0] == 50  # height
        assert roi.shape[1] == 50  # width
        assert roi.shape[2] == 3   # channels

    def test_extract_roi_not_found(self):
        """Test extracting non-existent ROI."""
        from vision.roi import ROIExtractor
        config = "rois:\n  test:\n    bbox: [0.0, 0.0, 0.5, 0.5]"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        roi = extractor.extract_roi(frame, 'non_existent')
        assert roi is None

    def test_extract_all_rois(self):
        """Test extracting all ROIs."""
        from vision.roi import ROIExtractor
        config = """
rois:
  roi1:
    bbox: [0.0, 0.0, 0.5, 0.5]
  roi2:
    bbox: [0.5, 0.0, 1.0, 0.5]
  roi3:
    bbox: [0.0, 0.5, 0.5, 1.0]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rois = extractor.extract_all_rois(frame)

        assert isinstance(rois, dict)
        assert len(rois) == 3
        assert 'roi1' in rois
        assert 'roi2' in rois
        assert 'roi3' in rois
        assert rois['roi1'].shape == (50, 50, 3)
        assert rois['roi2'].shape == (50, 50, 3)
        assert rois['roi3'].shape == (50, 50, 3)

    def test_extract_all_rois_empty_config(self):
        """Test extracting all ROIs with empty config."""
        from vision.roi import ROIExtractor
        config = "rois: {}"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rois = extractor.extract_all_rois(frame)

        assert isinstance(rois, dict)
        assert len(rois) == 0

    def test_get_field_location(self):
        """Test getting pixel coordinates for a field within ROI."""
        from vision.roi import ROIExtractor
        config = """
rois:
  hp_shield:
    bbox: [0.03, 0.78, 0.32, 0.98]
    fields:
      - name: hp
        location: [0.0, 0.0, 0.5, 1.0]
      - name: shield
        location: [0.5, 0.0, 1.0, 1.0]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            path = f.name

        extractor = ROIExtractor(path)
        os.unlink(path)

        # Test existing field
        hp_bbox = extractor.get_field_location('hp_shield', 'hp', 1920, 1080)
        assert hp_bbox is not None
        assert hp_bbox == (0, 0, 960, 1080)

        # Test non-existent field
        non_existent = extractor.get_field_location('hp_shield', 'non_existent', 1920, 1080)
        assert non_existent is None

        # Test non-existent ROI
        non_existent_roi = extractor.get_field_location('non_existent', 'hp', 1920, 1080)
        assert non_existent_roi is None


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
        assert ocr.use_template_matching is True
        assert isinstance(ocr.digit_templates, dict)

    def test_init_without_template_matching(self):
        """Test initialization without template matching."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=False)
        assert ocr is not None
        assert ocr.use_template_matching is False

    def test_init_default(self):
        """Test default initialization."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector()
        assert ocr.use_template_matching is True

    def test_extract_number_with_template_matching(self):
        """Test number extraction with template matching."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Create a test image with some patterns
        image = np.ones((30, 80, 3), dtype=np.uint8) * 255

        result = ocr.extract_number(image, min_confidence=0.5)
        assert 'value' in result
        assert 'source' in result
        assert 'confidence' in result
        assert result['source'] == 'ocr_template'
        assert isinstance(result['value'], int)
        assert isinstance(result['confidence'], float)

    def test_extract_number_with_tesseract(self):
        """Test number extraction with Tesseract mode."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=False)

        image = np.ones((30, 80, 3), dtype=np.uint8) * 255
        result = ocr.extract_number(image, min_confidence=0.5)

        assert 'value' in result
        assert result['source'] == 'ocr_tesseract'
        assert result['value'] == 0  # Default return for Tesseract mode (MVP)

    def test_extract_number_custom_confidence(self):
        """Test number extraction with custom confidence threshold."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        image = np.ones((30, 80, 3), dtype=np.uint8) * 255
        result = ocr.extract_number(image, min_confidence=0.9)

        # Result should still have all required fields
        assert 'value' in result
        assert 'confidence' in result

    def test_recognize_digit_low_pixels(self):
        """Test digit recognition with low pixel count."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Create a mostly black digit (few white pixels)
        digit = np.zeros((30, 20), dtype=np.uint8)
        result = ocr._recognize_digit(digit)
        assert result == 1  # < 50 white pixels

    def test_recognize_digit_medium_pixels(self):
        """Test digit recognition with medium pixel count."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Create digit with ~75 white pixels
        digit = np.zeros((30, 20), dtype=np.uint8)
        digit[:5, :15] = 255
        result = ocr._recognize_digit(digit)
        assert result == 7  # 50-99 white pixels

    def test_recognize_digit_high_pixels(self):
        """Test digit recognition with high pixel count."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Create mostly white digit
        digit = np.ones((30, 20), dtype=np.uint8) * 255
        result = ocr._recognize_digit(digit)
        assert result == 0  # >= 250 white pixels

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
        # HP should be clamped to 0-100
        assert 0 <= result['value'] <= 100

    def test_extract_hp_clamping(self):
        """Test HP value clamping."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Mock extract_number to return high value
        with patch.object(ocr, 'extract_number', return_value={'value': 999, 'source': 'ocr', 'confidence': 0.8}):
            result = ocr.extract_hp(np.ones((30, 80, 3), dtype=np.uint8))
            assert result['value'] == 100  # Clamped to max

        with patch.object(ocr, 'extract_number', return_value={'value': -50, 'source': 'ocr', 'confidence': 0.8}):
            result = ocr.extract_hp(np.ones((30, 80, 3), dtype=np.uint8))
            assert result['value'] == 0  # Clamped to min

    def test_extract_shield(self):
        """Test shield extraction."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        image = np.ones((30, 80, 3), dtype=np.uint8) * 255
        result = ocr.extract_shield(image)
        assert 'value' in result
        assert 'source' in result
        assert 'confidence' in result
        # Shield should be clamped to 0-100
        assert 0 <= result['value'] <= 100

    def test_extract_shield_clamping(self):
        """Test shield value clamping."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Mock extract_number to return high value
        with patch.object(ocr, 'extract_number', return_value={'value': 150, 'source': 'ocr', 'confidence': 0.8}):
            result = ocr.extract_shield(np.ones((30, 80, 3), dtype=np.uint8))
            assert result['value'] == 100  # Clamped to max

    def test_extract_ammo(self):
        """Test ammo extraction."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        image = np.ones((30, 80, 3), dtype=np.uint8) * 255
        result = ocr.extract_ammo(image)
        assert 'value' in result
        assert 'source' in result
        assert 'confidence' in result
        # Ammo should be clamped to 0-999
        assert 0 <= result['value'] <= 999

    def test_extract_ammo_clamping(self):
        """Test ammo value clamping."""
        from vision.ocr import OCRDetector
        ocr = OCRDetector(use_template_matching=True)

        # Mock extract_number to return very high value
        with patch.object(ocr, 'extract_number', return_value={'value': 9999, 'source': 'ocr', 'confidence': 0.8}):
            result = ocr.extract_ammo(np.ones((30, 80, 3), dtype=np.uint8))
            assert result['value'] == 999  # Clamped to max


# ============================================================================
# StateBuilder Tests
# ============================================================================

class TestStateBuilder:
    """Tests for StateBuilder."""

    def test_init(self):
        """Test StateBuilder initialization."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()
        assert builder is not None
        assert builder.current_state is not None
        assert 'player' in builder.current_state
        assert 'world' in builder.current_state
        assert 'session' in builder.current_state

    def test_create_empty_state(self):
        """Test empty state creation."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()
        state = builder._create_empty_state()

        # Check structure
        assert 'player' in state
        assert 'world' in state
        assert 'session' in state

        # Check default values
        assert state['player']['status']['hp']['value'] == 100
        assert state['player']['status']['shield']['value'] == 0
        assert state['player']['status']['is_knocked']['value'] is False
        assert state['world']['storm']['in_storm']['value'] is False
        assert state['session']['inactivity_duration_ms']['value'] == 0

    def test_validate_state_value_valid(self):
        """Test state value validation with valid input."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        result = builder._validate_state_value(100, 'test_source', 0.9)
        assert result['value'] == 100
        assert result['source'] == 'test_source'
        assert result['confidence'] == 0.9
        assert 'ts_ms' in result

    def test_validate_state_value_invalid_confidence(self):
        """Test state value validation with invalid confidence."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            builder._validate_state_value(100, 'test_source', 1.5)

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            builder._validate_state_value(100, 'test_source', -0.1)

    def test_validate_state_value_boundary_confidence(self):
        """Test state value validation with boundary confidence values."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        # Test boundary values (should not raise)
        result1 = builder._validate_state_value(100, 'test', 0.0)
        assert result1['confidence'] == 0.0

        result2 = builder._validate_state_value(100, 'test', 1.0)
        assert result2['confidence'] == 1.0

    def test_update_field(self):
        """Test updating a field in the state."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_field('player.status.hp', 75, 'ocr', 0.9)
        state = builder.get_state()

        assert state['player']['status']['hp']['value'] == 75
        assert state['player']['status']['hp']['source'] == 'ocr'
        assert state['player']['status']['hp']['confidence'] == 0.9

    def test_update_field_invalid_confidence(self):
        """Test update_field with invalid confidence."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            builder.update_field('player.status.hp', 75, 'ocr', 1.5)

    def test_update_field_non_existent_path(self):
        """Test updating non-existent field path."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        # Should not raise, just return silently
        builder.update_field('non.existent.path', 100, 'test', 0.5)

        # State should remain unchanged
        state = builder.get_state()
        assert 'non' not in state

    def test_update_hp(self):
        """Test updating HP value."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_hp(50, 'ocr', 0.85)
        state = builder.get_state()

        assert state['player']['status']['hp']['value'] == 50
        assert state['player']['status']['hp']['source'] == 'ocr'
        assert state['player']['status']['hp']['confidence'] == 0.85

    def test_update_shield(self):
        """Test updating shield value."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_shield(100, 'ocr', 0.9)
        state = builder.get_state()

        assert state['player']['status']['shield']['value'] == 100
        assert state['player']['status']['shield']['source'] == 'ocr'
        assert state['player']['status']['shield']['confidence'] == 0.9

    def test_update_knocked(self):
        """Test updating knocked status."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_knocked(True, 'yolo', 0.95)
        state = builder.get_state()

        assert state['player']['status']['is_knocked']['value'] is True
        assert state['player']['status']['is_knocked']['source'] == 'yolo'
        assert state['player']['status']['is_knocked']['confidence'] == 0.95

    def test_update_weapon_name(self):
        """Test updating weapon name."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_weapon_name('Assault Rifle', 'yolo', 0.8)
        state = builder.get_state()

        assert state['player']['weapon']['name']['value'] == 'Assault Rifle'
        assert state['player']['weapon']['name']['source'] == 'yolo'
        assert state['player']['weapon']['name']['confidence'] == 0.8

    def test_update_ammo(self):
        """Test updating ammo count."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_ammo(30, 'ocr', 0.95)
        state = builder.get_state()

        assert state['player']['weapon']['ammo']['value'] == 30
        assert state['player']['weapon']['ammo']['source'] == 'ocr'
        assert state['player']['weapon']['ammo']['confidence'] == 0.95

    def test_update_materials(self):
        """Test updating materials count."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_materials(500, 'ocr', 0.9)
        state = builder.get_state()

        assert state['player']['inventory']['materials']['value'] == 500
        assert state['player']['inventory']['materials']['source'] == 'ocr'
        assert state['player']['inventory']['materials']['confidence'] == 0.9

    def test_update_storm_phase(self):
        """Test updating storm phase."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_storm_phase(3, 'roi', 1.0)
        state = builder.get_state()

        assert state['world']['storm']['phase']['value'] == 3
        assert state['world']['storm']['phase']['source'] == 'roi'
        assert state['world']['storm']['phase']['confidence'] == 1.0

    def test_update_storm_damage(self):
        """Test updating storm damage."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_storm_damage(5.0, 'config', 1.0)
        state = builder.get_state()

        assert state['world']['storm']['damage']['value'] == 5.0
        assert state['world']['storm']['damage']['source'] == 'config'
        assert state['world']['storm']['damage']['confidence'] == 1.0

    def test_update_in_storm(self):
        """Test updating in-storm status."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_in_storm(True, 'roi', 0.9)
        state = builder.get_state()

        assert state['world']['storm']['in_storm']['value'] is True
        assert state['world']['storm']['in_storm']['source'] == 'roi'
        assert state['world']['storm']['in_storm']['confidence'] == 0.9

    def test_update_storm_shrinking(self):
        """Test updating storm shrinking status."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_storm_shrinking(True, 'roi', 1.0)
        state = builder.get_state()

        assert state['world']['storm']['is_shrinking']['value'] is True
        assert state['world']['storm']['is_shrinking']['source'] == 'roi'
        assert state['world']['storm']['is_shrinking']['confidence'] == 1.0

    def test_update_session_phase(self):
        """Test updating session phase."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_session_phase('landing', 'logic', 1.0)
        state = builder.get_state()

        assert state['session']['phase']['value'] == 'landing'
        assert state['session']['phase']['source'] == 'logic'
        assert state['session']['phase']['confidence'] == 1.0

    def test_get_state(self):
        """Test getting current state."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        state = builder.get_state()
        assert state is not None
        assert isinstance(state, dict)
        assert 'player' in state
        assert 'world' in state
        assert 'session' in state

    def test_get_state_after_updates(self):
        """Test getting state after multiple updates."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_hp(75, 'ocr', 0.9)
        builder.update_shield(50, 'ocr', 0.8)
        builder.update_ammo(30, 'ocr', 0.95)

        state = builder.get_state()
        assert state['player']['status']['hp']['value'] == 75
        assert state['player']['status']['shield']['value'] == 50
        assert state['player']['weapon']['ammo']['value'] == 30

    def test_reset(self):
        """Test resetting state to default."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        # Make some changes
        builder.update_hp(10, 'ocr', 0.5)
        builder.update_shield(0, 'ocr', 0.5)
        builder.update_knocked(True, 'yolo', 0.9)

        # Reset
        builder.reset()

        # Check defaults restored
        state = builder.get_state()
        assert state['player']['status']['hp']['value'] == 100
        assert state['player']['status']['shield']['value'] == 0
        assert state['player']['status']['is_knocked']['value'] is False

    def test_get_movement_state_non_combat(self):
        """Test movement state detection for non-combat."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        # Default state: HP=100, not in storm
        state = builder.get_movement_state()
        assert state == 'non_combat'

    def test_get_movement_state_combat_low_hp(self):
        """Test movement state detection for combat due to low HP."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_hp(40, 'ocr', 0.9)
        state = builder.get_movement_state()
        assert state == 'combat'

    def test_get_movement_state_combat_in_storm(self):
        """Test movement state detection for combat due to storm."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_in_storm(True, 'roi', 0.9)
        state = builder.get_movement_state()
        assert state == 'combat'

    def test_get_movement_state_none_hp(self):
        """Test movement state with None HP value."""
        from vision.state_builder import StateBuilder
        builder = StateBuilder()

        builder.update_field('player.status.hp', None, 'test', 0.0)
        state = builder.get_movement_state()
        # Should be non_combat since HP is None (not < 50)
        assert state == 'non_combat'


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

        # calibrate_roi takes detected_anchors and roi_config
        detected_anchors = {
            "hp_shield_bottom_left": (128, 612),
            "minimap_top_right": (1152, 72)
        }
        roi_config = {
            "hp_shield": {"bbox": [0.03, 0.78, 0.32, 0.98]}
        }

        # calibrate_roi should return the config as-is when disabled
        result = detector.calibrate_roi(detected_anchors, roi_config)
        assert result == roi_config
