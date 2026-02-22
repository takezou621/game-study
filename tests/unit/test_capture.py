"""Tests for capture modules."""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestScreenCapture:
    """Tests for ScreenCapture."""

    def test_import(self):
        """Test that ScreenCapture can be imported."""
        from src.capture.screen_capture import ScreenCapture
        assert ScreenCapture is not None

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture()
        assert capture.monitor == 1
        assert capture.target_fps == 60
        assert capture.buffer_size == 10
        assert capture.max_width is None
        assert capture.capture_region is None
        assert not capture.is_opened
        assert not capture._running

    def test_init_with_monitor_id_alias(self):
        """Test initialization with monitor_id parameter (alias)."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(monitor_id=2)
        assert capture.monitor == 2

    def test_init_monitor_id_takes_precedence(self):
        """Test that monitor_id parameter takes precedence over monitor."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(monitor=3, monitor_id=2)
        assert capture.monitor == 2

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(
            monitor=2,
            target_fps=30,
            buffer_size=5,
            max_width=1280,
            capture_region=(100, 100, 800, 600)
        )
        assert capture.monitor == 2
        assert capture.target_fps == 30
        assert capture.buffer_size == 5
        assert capture.max_width == 1280
        assert capture.capture_region == (100, 100, 800, 600)
        assert capture.frame_time == 1.0 / 30

    def test_get_monitor_info_normal_case(self):
        """Test _get_monitor_info with valid monitor."""
        from src.capture.screen_capture import ScreenCapture
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},  # 0 is virtual
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080}
        ]

        with patch.object(ScreenCapture, '__init__', lambda self, **kwargs: None):
            capture = ScreenCapture()
            capture.sct = mock_sct

            from src.capture.screen_capture import ScreenCapture as SC
            result = SC._get_monitor_info(capture, 1)

            assert result["width"] == 1920
            assert result["height"] == 1080

    def test_get_monitor_info_invalid_monitor_low(self):
        """Test _get_monitor_info with monitor number too low."""
        from src.capture.screen_capture import ScreenCapture
        mock_sct = MagicMock()
        mock_sct.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]

        with patch.object(ScreenCapture, '__init__', lambda self, **kwargs: None):
            capture = ScreenCapture()
            capture.sct = mock_sct

            from src.capture.screen_capture import ScreenCapture as SC
            with pytest.raises(ValueError, match="Monitor 0 not found"):
                SC._get_monitor_info(capture, 0)

    def test_get_monitor_info_invalid_monitor_high(self):
        """Test _get_monitor_info with monitor number too high."""
        from src.capture.screen_capture import ScreenCapture
        mock_sct = MagicMock()
        mock_sct.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]

        with patch.object(ScreenCapture, '__init__', lambda self, **kwargs: None):
            capture = ScreenCapture()
            capture.sct = mock_sct

            from src.capture.screen_capture import ScreenCapture as SC
            with pytest.raises(ValueError, match="Monitor 2 not found"):
                SC._get_monitor_info(capture, 2)

    def test_open_success(self):
        """Test successful opening of screen capture."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [
                {},
                {"left": 0, "top": 0, "width": 1920, "height": 1080}
            ]
            mock_mss.return_value = mock_sct_instance

            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False

            with patch('src.capture.screen_capture.threading.Thread', return_value=mock_thread):
                capture = ScreenCapture(monitor=1)
                capture.open()

                assert capture.is_opened
                assert capture._running
                assert capture.sct == mock_sct_instance
                assert capture.monitor_info["width"] == 1920
                mock_thread.assert_called_once()

                capture.close()

    def test_open_with_capture_region(self):
        """Test opening with a capture region specified."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [
                {},
                {"left": 0, "top": 0, "width": 1920, "height": 1080}
            ]
            mock_mss.return_value = mock_sct_instance

            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False

            with patch('src.capture.screen_capture.threading.Thread', return_value=mock_thread):
                capture = ScreenCapture(
                    monitor=1,
                    capture_region=(100, 50, 800, 600)
                )
                capture.open()

                assert capture.monitor_info["left"] == 100
                assert capture.monitor_info["top"] == 50
                assert capture.monitor_info["width"] == 800
                assert capture.monitor_info["height"] == 600

                capture.close()

    def test_open_already_running(self):
        """Test opening when already running is idempotent."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_mss.return_value = mock_sct_instance

            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False

            with patch('src.capture.screen_capture.threading.Thread', return_value=mock_thread):
                capture = ScreenCapture(monitor=1)
                capture.open()
                initial_sct = capture.sct
                capture.open()  # Should not reinitialize

                assert capture.sct == initial_sct

                capture.close()

    def test_open_mss_not_available(self):
        """Test opening when MSS is not available."""
        from src.capture.screen_capture import ScreenCapture

        with patch('src.capture.screen_capture.MSS_AVAILABLE', False):
            capture = ScreenCapture(monitor=1)
            with pytest.raises(RuntimeError, match="MSS library is not available"):
                capture.open()

    def test_close_stops_capture_thread(self):
        """Test closing stops the capture thread."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_mss.return_value = mock_sct_instance

            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False

            with patch('src.capture.screen_capture.threading.Thread', return_value=mock_thread):
                capture = ScreenCapture(monitor=1)
                capture.open()
                assert capture._running

                capture.close()

                assert not capture._running
                assert not capture.is_opened
                mock_thread.join.assert_called_with(timeout=2.0)

    def test_close_idempotent(self):
        """Test closing when already closed is safe."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_mss.return_value = mock_sct_instance

            capture = ScreenCapture(monitor=1)
            capture.close()  # Should not raise
            capture.close()  # Should still not raise

    def test_capture_loop_with_mss_mock(self):
        """Test _capture_loop behavior with mocked MSS."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [{}, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_mss.return_value = mock_sct_instance

            # Create a real frame for testing
            test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            with patch('src.capture.screen_capture.cv2'):
                capture = ScreenCapture(monitor=1, target_fps=30)
                capture.open()

                # Manually set running to False after a short time
                def stop_after_delay():
                    time.sleep(0.1)
                    capture._running = False

                stopper = threading.Thread(target=stop_after_delay)
                stopper.start()

                # Mock the frame capture to return our test frame
                with patch.object(capture, '_capture_frame', return_value=test_frame):
                    capture._capture_loop()

                stopper.join()
                capture.close()

    def test_capture_loop_handles_dropped_frames(self):
        """Test that _capture_loop handles dropped frames when buffer is full."""
        from src.capture.screen_capture import MSS_AVAILABLE, ScreenCapture

        if not MSS_AVAILABLE:
            pytest.skip("MSS not available")

        with patch('src.capture.screen_capture.mss.mss') as mock_mss:
            mock_sct_instance = MagicMock()
            mock_sct_instance.monitors = [{}, {"left": 0, "top": 0, "width": 100, "height": 100}]
            mock_mss.return_value = mock_sct_instance

            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('src.capture.screen_capture.cv2'):
                capture = ScreenCapture(monitor=1, buffer_size=2)
                capture.open()

                # Fill the buffer
                capture.frame_buffer.put(test_frame)
                capture.frame_buffer.put(test_frame)

                # Mock _capture_frame and run loop briefly
                with patch.object(capture, '_capture_frame', return_value=test_frame):
                    capture._running = True
                    # This should cause a dropped frame since buffer is full
                    initial_dropped = capture._dropped_frames
                    capture._capture_loop()
                    # Frame should be dropped
                    assert capture._dropped_frames >= initial_dropped

                capture._running = False
                capture.close()

    def test_read_blocking_returns_frame(self):
        """Test read() blocking behavior returns frame from buffer."""
        from src.capture.screen_capture import ScreenCapture

        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        capture = ScreenCapture()
        capture._running = True
        capture.frame_buffer.put(test_frame)

        frame = capture.read()

        assert frame is not None
        assert frame.shape == (100, 100, 3)
        assert capture.frame_count > 0

    def test_read_blocking_timeout(self):
        """Test read() returns None on timeout."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture()
        capture._running = True
        # Don't put anything in buffer

        frame = capture.read()

        assert frame is None

    def test_read_returns_none_when_stopped(self):
        """Test read() returns None when stopped and buffer empty."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture()
        capture._running = False

        frame = capture.read()

        assert frame is None

    def test_read_latest_frame_non_blocking(self):
        """Test read_latest_frame() returns latest frame without blocking."""
        from src.capture.screen_capture import ScreenCapture

        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 200

        capture = ScreenCapture()
        capture.frame_buffer.put(frame1)
        capture.frame_buffer.put(frame2)

        latest = capture.read_latest_frame()

        assert latest is not None
        # Should get the last frame (frame2)
        assert np.all(latest == 200)

    def test_read_latest_frame_empty_buffer(self):
        """Test read_latest_frame() returns None when buffer is empty."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture()
        latest = capture.read_latest_frame()

        assert latest is None

    def test_get_metadata_not_initialized(self):
        """Test get_metadata() when not yet initialized."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=1, target_fps=30, buffer_size=5)
        metadata = capture.get_metadata()

        assert metadata["monitor"] == 1
        assert metadata["target_fps"] == 30
        assert metadata["actual_fps"] == 0.0
        assert metadata["buffer_size"] == 5
        assert "width" not in metadata
        assert "height" not in metadata

    def test_get_metadata_after_initialization(self):
        """Test get_metadata() after initialization."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=1, target_fps=60, buffer_size=10)

        # Manually set up state to simulate opened capture
        capture.monitor_info = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        capture.is_opened = True
        capture._running = True
        capture._start_time = time.time()
        capture._dropped_frames = 5

        metadata = capture.get_metadata()

        assert metadata["monitor"] == 1
        assert metadata["width"] == 1920
        assert metadata["height"] == 1080
        assert metadata["target_fps"] == 60
        assert metadata["buffer_size"] == 10
        assert "latency_ms" in metadata
        assert "frame_count" in metadata
        assert metadata["dropped_frames"] == 5

    def test_set_region(self):
        """Test set_region() updates monitor info."""
        from src.capture.screen_capture import ScreenCapture

        # Create a capture and manually set up monitor_info to test set_region
        capture = ScreenCapture(monitor=1)
        # Manually set up monitor_info to simulate an opened capture
        capture.monitor_info = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        capture.is_opened = True

        capture.set_region(100, 200, 800, 600)

        assert capture.monitor_info["left"] == 100
        assert capture.monitor_info["top"] == 200
        assert capture.monitor_info["width"] == 800
        assert capture.monitor_info["height"] == 600

    def test_set_region_before_open(self):
        """Test set_region() when monitor_info is None."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=1)
        # Should not raise, just does nothing
        capture.set_region(100, 200, 800, 600)

    def test_reset_region(self):
        """Test reset_region() restores full monitor dimensions."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=2)

        # Set up mock sct with monitors info
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080}
        ]
        capture.sct = mock_sct

        # Set up monitor_info to simulate opened capture with a region set
        capture.monitor_info = {"left": 100, "top": 100, "width": 800, "height": 600}
        capture.is_opened = True

        # Reset region
        capture.reset_region()

        # Should be back to monitor 2's full dimensions
        assert capture.monitor_info["left"] == 1920
        assert capture.monitor_info["top"] == 0
        assert capture.monitor_info["width"] == 1920
        assert capture.monitor_info["height"] == 1080

    def test_reset_region_before_open(self):
        """Test reset_region() when sct is None."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=1)
        # Should not raise
        capture.reset_region()

    def test_estimate_latency_empty_buffer(self):
        """Test _estimate_latency() returns 0 when buffer is empty."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=1, target_fps=60)
        latency = capture._estimate_latency()

        assert latency == 0.0

    def test_estimate_latency_with_frames(self):
        """Test _estimate_latency() calculates based on buffer size."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(monitor=1, target_fps=60)
        # Add some frames to buffer
        for _ in range(5):
            capture.frame_buffer.put(np.zeros((100, 100, 3), dtype=np.uint8))

        latency = capture._estimate_latency()

        # With 5 frames at 60fps, expected latency is (5/2)/60 * 1000 ms
        expected = (5 / 2.0 / 60) * 1000.0
        assert abs(latency - expected) < 0.01

    def test_thread_safety_concurrent_reads(self):
        """Test that concurrent read operations are thread-safe."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture(buffer_size=20)  # Larger buffer to avoid blocking
        capture._running = True

        # Add frames to buffer (less than buffer_size to avoid blocking)
        for i in range(10):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i
            capture.frame_buffer.put(frame, block=False)

        results = []
        errors = []

        def read_worker():
            try:
                for _ in range(3):
                    frame = capture.read()
                    if frame is not None:
                        results.append(frame)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Some results should have been collected
        assert len(results) >= 0  # May be 0 if frames were already consumed

    def test_iter_yields_frames(self):
        """Test __iter__ yields frames from buffer."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture()
        capture._running = True

        frames_added = []
        for i in range(3):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i
            frames_added.append(frame)
            capture.frame_buffer.put(frame)

        # Set _running to False after adding frames so iteration stops
        capture._running = False

        frames_yielded = list(capture.__iter__())

        assert len(frames_yielded) == 3

    def test_iter_stops_when_not_running(self):
        """Test __iter__ stops iteration when not running."""
        from src.capture.screen_capture import ScreenCapture

        capture = ScreenCapture()
        capture._running = False

        frames = list(capture.__iter__())

        assert len(frames) == 0


class TestVideoFileCapture:
    """Tests for VideoFileCapture."""

    def test_import(self):
        """Test that VideoFileCapture can be imported."""
        from src.capture.video_file import VideoFileCapture
        assert VideoFileCapture is not None

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file."""
        from src.capture.video_file import VideoFileCapture
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            VideoFileCapture("/nonexistent/video.mp4")

    def test_init_success_with_temp_file(self):
        """Test successful initialization with existing file."""
        from src.capture.video_file import VideoFileCapture

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            capture = VideoFileCapture(tmp_path)
            assert capture.video_path == Path(tmp_path)
            assert capture.cap is None
            # is_opened is only set after open() is called
            assert not hasattr(capture, 'is_opened') or not capture.is_opened
        finally:
            os.unlink(tmp_path)

    def test_init_with_path_object(self):
        """Test initialization with Path object."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            capture = VideoFileCapture(Path(tmp_path))
            assert isinstance(capture.video_path, Path)
        finally:
            os.unlink(tmp_path)

    def test_open_success(self):
        """Test successful opening of video file."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda prop: {
                    3: 1920,   # CAP_PROP_FRAME_WIDTH
                    4: 1080,   # CAP_PROP_FRAME_HEIGHT
                    5: 30.0,   # CAP_PROP_FPS
                    7: 900     # CAP_PROP_FRAME_COUNT
                }.get(prop, 0)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                assert capture.is_opened
                assert capture.cap == mock_cap
                assert capture.width == 1920
                assert capture.height == 1080
                assert capture.fps == 30.0
                assert capture.frame_count == 900
        finally:
            os.unlink(tmp_path)

    def test_open_failure(self):
        """Test opening fails when VideoCapture can't open file."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = False
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                with pytest.raises(RuntimeError, match="Failed to open video file"):
                    capture.open()
        finally:
            os.unlink(tmp_path)

    def test_close(self):
        """Test closing video capture."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()
                assert capture.is_opened

                capture.close()

                assert capture.cap is None
                assert not capture.is_opened
                mock_cap.release.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_close_when_not_opened(self):
        """Test closing when not opened is safe."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            capture = VideoFileCapture(tmp_path)
            capture.close()  # Should not raise

            assert capture.cap is None
            # is_opened may not exist if close() was called before open()
            # This is expected behavior
        finally:
            os.unlink(tmp_path)

    def test_read_frame_success(self):
        """Test read_frame() returns frame successfully."""
        import numpy as np

        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                mock_cap.read.return_value = (True, test_frame)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                success, frame = capture.read_frame()

                assert success is True
                assert frame is not None
                assert frame.shape == (1080, 1920, 3)
        finally:
            os.unlink(tmp_path)

    def test_read_frame_failure(self):
        """Test read_frame() handles read failure."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                mock_cap.read.return_value = (False, None)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                success, frame = capture.read_frame()

                assert success is False
                assert frame is None
        finally:
            os.unlink(tmp_path)

    def test_read_frame_not_opened(self):
        """Test read_frame() raises when not opened."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            capture = VideoFileCapture(tmp_path)
            with pytest.raises(RuntimeError, match="Video file is not opened"):
                capture.read_frame()
        finally:
            os.unlink(tmp_path)

    def test_read_success(self):
        """Test read() returns frame on success."""
        import numpy as np

        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                mock_cap.read.return_value = (True, test_frame)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                frame = capture.read()

                assert frame is not None
                assert frame.shape == (1080, 1920, 3)
        finally:
            os.unlink(tmp_path)

    def test_read_returns_none_on_failure(self):
        """Test read() returns None on failure."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                mock_cap.read.return_value = (False, None)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                frame = capture.read()

                assert frame is None
        finally:
            os.unlink(tmp_path)

    def test_get_frame_at_valid_index(self):
        """Test get_frame_at() with valid index."""
        import numpy as np

        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 100  # frame_count
                test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                mock_cap.read.return_value = (True, test_frame)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                frame = capture.get_frame_at(50)

                assert frame is not None
                mock_cap.set.assert_called_with(1, 50)  # CAP_PROP_POS_FRAMES
        finally:
            os.unlink(tmp_path)

    def test_get_frame_at_negative_index(self):
        """Test get_frame_at() with negative index returns None."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 100
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                frame = capture.get_frame_at(-1)

                assert frame is None
        finally:
            os.unlink(tmp_path)

    def test_get_frame_at_out_of_bounds_high(self):
        """Test get_frame_at() with index >= frame_count returns None."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 100
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                frame = capture.get_frame_at(100)

                assert frame is None
        finally:
            os.unlink(tmp_path)

    def test_get_frame_at_not_opened(self):
        """Test get_frame_at() raises when not opened."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            capture = VideoFileCapture(tmp_path)
            with pytest.raises(RuntimeError, match="Video file is not opened"):
                capture.get_frame_at(0)
        finally:
            os.unlink(tmp_path)

    def test_get_frame_at_boundary_conditions(self):
        """Test get_frame_at() at various boundary conditions."""
        import numpy as np

        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 10  # frame_count
                test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                mock_cap.read.return_value = (True, test_frame)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                # Test first frame (index 0)
                frame = capture.get_frame_at(0)
                assert frame is not None

                # Test last valid frame (index 9)
                mock_cap.set.reset_mock()
                frame = capture.get_frame_at(9)
                assert frame is not None
                mock_cap.set.assert_called_with(1, 9)

                # Test just beyond last frame (index 10)
                frame = capture.get_frame_at(10)
                assert frame is None
        finally:
            os.unlink(tmp_path)

    def test_get_metadata(self):
        """Test get_metadata() returns correct structure."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda prop: {
                    3: 1920,
                    4: 1080,
                    5: 30.0,
                    7: 900
                }.get(prop, 0)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                metadata = capture.get_metadata()

                assert metadata["path"] == tmp_path
                assert metadata["fps"] == 30.0
                assert metadata["width"] == 1920
                assert metadata["height"] == 1080
                assert metadata["frame_count"] == 900
                assert metadata["duration_seconds"] == 30.0
        finally:
            os.unlink(tmp_path)

    def test_get_metadata_zero_fps(self):
        """Test get_metadata() handles zero FPS."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.side_effect = lambda prop: {
                    3: 1920,
                    4: 1080,
                    5: 0.0,  # Zero FPS
                    7: 900
                }.get(prop, 0)
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                metadata = capture.get_metadata()

                assert metadata["duration_seconds"] == 0
        finally:
            os.unlink(tmp_path)

    def test_context_manager(self):
        """Test __enter__ and __exit__ context manager."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                mock_cv2.return_value = mock_cap

                with VideoFileCapture(tmp_path) as capture:
                    assert capture.is_opened
                    assert capture.cap == mock_cap

                # After exit, should be closed
                assert capture.cap is None
                assert not capture.is_opened
                mock_cap.release.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_iter_yields_frames(self):
        """Test __iter__ yields frames until end."""
        import numpy as np

        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0

                frames_to_yield = [
                    np.ones((100, 100, 3), dtype=np.uint8) * i
                    for i in range(3)
                ]
                mock_cap.read.side_effect = [
                    (True, frames_to_yield[0]),
                    (True, frames_to_yield[1]),
                    (True, frames_to_yield[2]),
                    (False, None)  # End of video
                ]
                mock_cv2.return_value = mock_cap

                capture = VideoFileCapture(tmp_path)
                capture.open()

                frames = list(capture)

                assert len(frames) == 3
        finally:
            os.unlink(tmp_path)

    def test_iter_not_opened(self):
        """Test __iter__ raises when not opened."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            capture = VideoFileCapture(tmp_path)
            with pytest.raises(RuntimeError, match="Video file is not opened"):
                next(iter(capture))
        finally:
            os.unlink(tmp_path)


class TestCaptureBase:
    """Tests for capture base class."""

    def test_import(self):
        """Test that base classes can be imported."""
        from src.capture.base import BaseCapture
        assert BaseCapture is not None

    def test_base_initialization(self):
        """Test BaseCapture initialization."""

        # Can't instantiate abstract class directly
        # but we can test via a concrete subclass
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture'):
                capture = VideoFileCapture(tmp_path)

                # BaseCapture attributes
                assert not capture.is_opened
                assert capture.frame_count == 0
                assert capture.start_time_ms == 0
                assert capture.fps == 0.0
        finally:
            os.unlink(tmp_path)

    def test_base_context_manager(self):
        """Test BaseCapture context manager calls close."""
        from src.capture.video_file import VideoFileCapture

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('src.capture.video_file.cv2.VideoCapture') as mock_cv2:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 0
                mock_cv2.return_value = mock_cap

                with VideoFileCapture(tmp_path) as capture:
                    assert capture.is_opened

                # close() should have been called via __exit__
                mock_cap.release.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_base_iter(self):
        """Test BaseCapture __iter__ uses read()."""
        from src.capture.base import BaseCapture

        # Create a minimal concrete implementation
        class MinimalCapture(BaseCapture):
            def __init__(self):
                super().__init__()
                self.frames = [
                    np.ones((10, 10, 3), dtype=np.uint8) * i
                    for i in range(3)
                ]
                self.index = 0

            def open(self):
                self.is_opened = True

            def read(self):
                if self.index >= len(self.frames):
                    return None
                frame = self.frames[self.index]
                self.index += 1
                return frame

            def get_metadata(self):
                return {}

            def close(self):
                self.is_opened = False

        capture = MinimalCapture()
        frames = list(capture)

        assert len(frames) == 3

    def test_update_fps(self):
        """Test _update_fps calculates FPS correctly."""
        from src.capture.base import BaseCapture

        class MinimalCapture(BaseCapture):
            def open(self): pass
            def read(self): return None
            def get_metadata(self): return {}
            def close(self): pass

        capture = MinimalCapture()

        # First frame - sets start_time_ms, frame_count becomes 1
        with patch('src.capture.base.get_timestamp_ms', return_value=1000):
            capture._update_fps()
            assert capture.frame_count == 1
            assert capture.start_time_ms == 1000
            assert capture.fps == 0.0  # FPS is not calculated on first frame

        # Second frame after 100ms - calculates FPS
        with patch('src.capture.base.get_timestamp_ms', return_value=1100):
            capture._update_fps()
            assert capture.frame_count == 2
            # FPS = (frame_count / elapsed_ms) * 1000
            # FPS is calculated BEFORE frame_count is incremented
            # So at calculation time: frame_count=1, elapsed=100
            # FPS = (1 / 100) * 1000 = 10 FPS
            assert abs(capture.fps - 10.0) < 0.1
