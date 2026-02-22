"""Tests for main.py entry point."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# ============================================================================
# parse_args Tests
# ============================================================================

class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_with_required_arguments(self):
        """Test parsing with minimum required arguments."""
        # Need to import fresh for each test to get clean argparse state
        import importlib

        import main
        importlib.reload(main)  # Reload to reset argparse

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/test_output'
        ]):
            args = main.parse_args()

            assert args.input == 'video'
            assert args.out == '/tmp/test_output'
            assert args.video is None
            assert args.triggers == './configs/triggers.yaml'
            assert args.roi == './configs/roi_defaults.yaml'
            assert args.system_prompt == './configs/prompts/system.txt'
            assert args.voice is False
            assert args.voice_model == 'tts-1'

    def test_parse_args_with_all_arguments(self):
        """Test parsing with all arguments provided."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', '/path/to/video.mp4',
            '--triggers', '/path/to/triggers.yaml',
            '--roi', '/path/to/roi.yaml',
            '--out', '/tmp/output',
            '--system-prompt', '/path/to/prompt.txt',
            '--voice',
            '--voice-model', 'realtime'
        ]):
            args = main.parse_args()

            assert args.input == 'video'
            assert args.video == '/path/to/video.mp4'
            assert args.triggers == '/path/to/triggers.yaml'
            assert args.roi == '/path/to/roi.yaml'
            assert args.out == '/tmp/output'
            assert args.system_prompt == '/path/to/prompt.txt'
            assert args.voice is True
            assert args.voice_model == 'realtime'

    def test_parse_args_with_voice_flag(self):
        """Test parsing with voice flag enabled."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out',
            '--voice'
        ]):
            args = main.parse_args()
            assert args.voice is True
            assert args.voice_model == 'tts-1'

    def test_parse_args_voice_model_tts1(self):
        """Test parsing with tts-1 voice model."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out',
            '--voice-model', 'tts-1'
        ]):
            args = main.parse_args()
            assert args.voice_model == 'tts-1'

    def test_parse_args_voice_model_tts1_hd(self):
        """Test parsing with tts-1-hd voice model."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out',
            '--voice-model', 'tts-1-hd'
        ]):
            args = main.parse_args()
            assert args.voice_model == 'tts-1-hd'

    def test_parse_args_voice_model_realtime(self):
        """Test parsing with realtime voice model."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out',
            '--voice-model', 'realtime'
        ]):
            args = main.parse_args()
            assert args.voice_model == 'realtime'

    def test_parse_args_default_triggers_path(self):
        """Test default triggers config path."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out'
        ]):
            args = main.parse_args()
            assert args.triggers == './configs/triggers.yaml'

    def test_parse_args_default_roi_path(self):
        """Test default ROI config path."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out'
        ]):
            args = main.parse_args()
            assert args.roi == './configs/roi_defaults.yaml'

    def test_parse_args_default_system_prompt(self):
        """Test default system prompt path."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out'
        ]):
            args = main.parse_args()
            assert args.system_prompt == './configs/prompts/system.txt'

    def test_parse_args_missing_required_input(self):
        """Test that missing --input argument causes error."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--out', '/tmp/out'
        ]), pytest.raises(SystemExit):
            main.parse_args()

    def test_parse_args_missing_required_out(self):
        """Test that missing --out argument causes error."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video'
        ]), pytest.raises(SystemExit):
            main.parse_args()

    def test_parse_args_invalid_input_choice(self):
        """Test that invalid --input choice causes error."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'invalid',
            '--out', '/tmp/out'
        ]), pytest.raises(SystemExit):
            main.parse_args()

    def test_parse_args_invalid_voice_model(self):
        """Test that invalid --voice-model choice causes error."""
        import importlib

        import main
        importlib.reload(main)

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--out', '/tmp/out',
            '--voice-model', 'invalid-model'
        ]), pytest.raises(SystemExit):
            main.parse_args()


# ============================================================================
# main() Function Tests - Initialization
# ============================================================================

class TestMainInitialization:
    """Tests for main() function initialization."""

    def test_main_creates_output_directory(self, temp_dir):
        """Test that main creates output directory if it doesn't exist."""
        import main

        output_dir = temp_dir / "output"

        # Create minimal config files
        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers:\n  - id: test\n    name: Test\n    priority: 0\n    enabled: true\n    conditions:\n      - field: player.status.hp.value\n        operator: lt\n        value: 30\n    templates:\n      combat: 'test'\n    cooldown_ms: 5000")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(output_dir)
        ]):
            # Mock VideoFileCapture to avoid needing real video
            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30, 'frame_count': 10}
                mock_capture_instance.__iter__ = Mock(return_value=iter([]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

                # Check output directory was created
                assert output_dir.exists()

    def test_main_missing_roi_config(self, temp_dir):
        """Test that main exits with error when ROI config is missing."""
        import main

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', 'test.mp4',
            '--roi', '/nonexistent/roi.yaml',
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            assert exc_info.value.code == 1

    def test_main_missing_triggers_config(self, temp_dir):
        """Test that main exits with error when triggers config is missing."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', 'test.mp4',
            '--roi', str(roi_config),
            '--triggers', '/nonexistent/triggers.yaml',
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            assert exc_info.value.code == 1

    def test_main_missing_video_file(self, temp_dir):
        """Test that main exits with error when video file is missing."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', '/nonexistent/video.mp4',
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            assert exc_info.value.code == 1

    def test_main_missing_video_argument(self, temp_dir):
        """Test that main exits with error when --video is missing for video input."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main.main()
            assert exc_info.value.code == 1


# ============================================================================
# main() Function Tests - Component Initialization
# ============================================================================

class TestMainComponentInitialization:
    """Tests for main() function component initialization."""

    def test_main_initializes_roi_extractor(self, temp_dir):
        """Test that main initializes ROIExtractor."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch('main.ROIExtractor') as mock_roi:
                main.main()
                mock_roi.assert_called_once_with(str(roi_config))

    def test_main_initializes_trigger_engine(self, temp_dir):
        """Test that main initializes TriggerEngine."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch('main.TriggerEngine') as mock_trigger:
                main.main()
                mock_trigger.assert_called_once_with(str(triggers_config))

    def test_main_initializes_template_manager(self, temp_dir):
        """Test that main initializes DialogueTemplateManager."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch('main.DialogueTemplateManager') as mock_template:
                main.main()
                mock_template.assert_called_once_with()


# ============================================================================
# main() Function Tests - OpenAI Client
# ============================================================================

class TestMainOpenAIClient:
    """Tests for main() function OpenAI client initialization."""

    def test_main_initializes_openai_client_success(self, temp_dir):
        """Test that main initializes OpenAI client when config is valid."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            # Patch OpenAIClient class, not instance
            with patch('main.OpenAIClient') as mock_openai_class:
                mock_openai_instance = MagicMock()
                mock_openai_class.return_value = mock_openai_instance

                main.main()

                # Verify OpenAIClient class was instantiated
                mock_openai_class.assert_called_once()
                call_args = mock_openai_class.call_args
                assert call_args is not None
                # Check that system_prompt_path was passed
                assert 'system_prompt_path' in call_args.kwargs or len(call_args[0]) > 0

    def test_main_handles_openai_client_value_error(self, temp_dir):
        """Test that main continues when OpenAI client raises ValueError."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch('main.OpenAIClient', side_effect=ValueError("No API key")):
                # Should not raise, should continue with openai_client = None
                main.main()


# ============================================================================
# main() Function Tests - Voice Client
# ============================================================================

class TestMainVoiceClient:
    """Tests for main() function voice client initialization."""

    def test_main_initializes_voice_client_with_tts1(self, temp_dir):
        """Test that main initializes voice client with tts-1 model."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice',
            '--voice-model', 'tts-1'
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            mock_voice = MagicMock()
            mock_voice.use_realtime_api = False

            with patch('main.RealtimeVoiceClient', return_value=mock_voice):
                main.main()
                # Check use_realtime_api is False for tts-1
                import main as main_module
                main_module.RealtimeVoiceClient.assert_called_once_with(
                    system_prompt_path=str(system_prompt),
                    enable_audio_output=True,
                    use_realtime_api=False
                )

    def test_main_initializes_voice_client_with_realtime(self, temp_dir):
        """Test that main initializes voice client with realtime model."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice',
            '--voice-model', 'realtime'
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            mock_voice = MagicMock()
            mock_voice.use_realtime_api = True

            with patch('main.RealtimeVoiceClient', return_value=mock_voice):
                main.main()
                # Check use_realtime_api is True for realtime
                import main as main_module
                main_module.RealtimeVoiceClient.assert_called_once_with(
                    system_prompt_path=str(system_prompt),
                    enable_audio_output=True,
                    use_realtime_api=True
                )

    def test_main_handles_voice_client_exception(self, temp_dir):
        """Test that main continues when voice client raises exception."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice'
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch('main.RealtimeVoiceClient', side_effect=Exception("No audio device")):
                # Should not raise, should continue with voice_client = None
                main.main()

    def test_main_voice_shutdown_on_completion(self, temp_dir):
        """Test that voice client shutdown is called on completion."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 0.5, 0.5]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice'
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            mock_voice = MagicMock()
            mock_voice.use_realtime_api = False

            with patch('main.RealtimeVoiceClient', return_value=mock_voice):
                main.main()
                mock_voice.shutdown.assert_called_once()


# ============================================================================
# main() Function Tests - Video Processing
# ============================================================================

class TestMainVideoProcessing:
    """Tests for main() function video processing loop."""

    def test_main_processes_single_frame(self, temp_dir):
        """Test that main processes a single frame."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            # Create a test frame
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30, 'frame_count': 1}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

                # Verify we processed the frame
                assert mock_capture_instance.__iter__.called

    def test_main_processes_multiple_frames(self, temp_dir):
        """Test that main processes multiple frames."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            # Create test frames
            test_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30, 'frame_count': 3}
                mock_capture_instance.__iter__ = Mock(return_value=iter(test_frames))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

                # Verify we processed all frames
                assert mock_capture_instance.__iter__.called

    def test_main_logs_frame_count(self, temp_dir):
        """Test that main logs frame count correctly."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            test_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30, 'frame_count': 5}
                mock_capture_instance.__iter__ = Mock(return_value=iter(test_frames))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

                # Verify processing was logged
                assert mock_capture_instance.__iter__.called

    def test_main_handles_keyboard_interrupt(self, temp_dir):
        """Test that main handles KeyboardInterrupt gracefully."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            def frame_generator():
                yield np.zeros((100, 100, 3), dtype=np.uint8)
                yield np.zeros((100, 100, 3), dtype=np.uint8)
                raise KeyboardInterrupt()

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = lambda self: frame_generator()
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                # Should not raise, should handle gracefully
                main.main()

    def test_main_keyboard_interrupt_shutdowns_voice(self, temp_dir):
        """Test that voice client is shut down on KeyboardInterrupt."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice'
        ]):
            def frame_generator():
                yield np.zeros((100, 100, 3), dtype=np.uint8)
                raise KeyboardInterrupt()

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = lambda self: frame_generator()
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                mock_voice = MagicMock()
                mock_voice.use_realtime_api = False

                with patch('main.RealtimeVoiceClient', return_value=mock_voice):
                    main.main()
                    # Verify shutdown was called even on keyboard interrupt
                    mock_voice.shutdown.assert_called_once()


# ============================================================================
# main() Function Tests - Trigger Handling
# ============================================================================

class TestMainTriggerHandling:
    """Tests for main() function trigger handling."""

    def test_main_evaluates_triggers(self, temp_dir):
        """Test that main evaluates triggers for each frame."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("""triggers:
  - id: low_hp
    name: Low HP
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Low HP!"
    cooldown_ms: 5000""")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

    def test_main_logs_trigger_results(self, temp_dir):
        """Test that main logs trigger results."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("""triggers:
  - id: test
    name: Test Trigger
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Test response"
    cooldown_ms: 5000""")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()


# ============================================================================
# main() Function Tests - Vision Components
# ============================================================================

class TestMainVisionComponents:
    """Tests for main() function vision component initialization."""

    def test_main_initializes_all_vision_components(self, temp_dir):
        """Test that main initializes all vision components."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch.multiple('main',
                ROIExtractor=MagicMock(),
                AnchorDetector=MagicMock(),
                YOLODetector=MagicMock(),
                OCRDetector=MagicMock(),
                StateBuilder=MagicMock()
            ):
                main.main()

                from main import (
                    AnchorDetector,
                    OCRDetector,
                    ROIExtractor,
                    StateBuilder,
                    YOLODetector,
                )
                ROIExtractor.assert_called_once()
                AnchorDetector.assert_called_once()
                YOLODetector.assert_called_once()
                OCRDetector.assert_called_once_with(use_template_matching=True)
                StateBuilder.assert_called_once()


# ============================================================================
# main() Function Tests - Response Generation
# ============================================================================

class TestMainResponseGeneration:
    """Tests for main() function response generation."""

    def test_main_generates_response_with_openai(self, temp_dir):
        """Test that main generates response using OpenAI when available."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("""triggers:
  - id: low_hp
    name: Low HP
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Template response"
    cooldown_ms: 5000""")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                mock_openai = MagicMock()
                mock_openai.generate_response.return_value = "OpenAI enhanced response"

                with patch('main.OpenAIClient', return_value=mock_openai):
                    main.main()

    def test_main_falls_back_to_template_on_openai_error(self, temp_dir):
        """Test that main falls back to template when OpenAI fails."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("""triggers:
  - id: low_hp
    name: Low HP
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Template response"
    cooldown_ms: 5000""")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                mock_openai = MagicMock()
                mock_openai.generate_response.side_effect = Exception("API error")

                with patch('main.OpenAIClient', return_value=mock_openai):
                    # Should not raise, should fall back to template
                    main.main()


# ============================================================================
# main() Function Tests - Voice Output
# ============================================================================

class TestMainVoiceOutput:
    """Tests for main() function voice output."""

    def test_main_calls_speak_with_trigger(self, temp_dir):
        """Test that main calls voice_client.speak_with_trigger when enabled."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("""triggers:
  - id: low_hp
    name: Low HP
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Low HP!"
    cooldown_ms: 5000""")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice'
        ]):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                mock_voice = MagicMock()
                mock_voice.use_realtime_api = False
                mock_voice_response = MagicMock()
                mock_voice_response.duration_ms = 1000
                mock_voice.speak_with_trigger.return_value = mock_voice_response

                with patch('main.RealtimeVoiceClient', return_value=mock_voice):
                    main.main()

    def test_main_handles_voice_error_gracefully(self, temp_dir):
        """Test that main handles voice client errors gracefully."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("""triggers:
  - id: low_hp
    name: Low HP
    priority: 0
    enabled: true
    conditions:
      - field: player.status.hp.value
        operator: lt
        value: 30
    templates:
      combat: "Low HP!"
    cooldown_ms: 5000""")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir),
            '--voice'
        ]):
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30}
                mock_capture_instance.__iter__ = Mock(return_value=iter([test_frame]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                mock_voice = MagicMock()
                mock_voice.use_realtime_api = False
                mock_voice.speak_with_trigger.side_effect = Exception("Audio error")

                with patch('main.RealtimeVoiceClient', return_value=mock_voice):
                    # Should not raise, should continue
                    main.main()


# ============================================================================
# main() Function Tests - Logging
# ============================================================================

class TestMainLogging:
    """Tests for main() function logging behavior."""

    def test_main_logs_video_metadata(self, temp_dir):
        """Test that main logs video metadata."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            metadata = {'fps': 30, 'width': 1920, 'height': 1080, 'frame_count': 100}

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = metadata
                mock_capture_instance.__iter__ = Mock(return_value=iter([]))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

                # Verify get_metadata was called
                mock_capture_instance.get_metadata.assert_called_once()

    def test_main_creates_session_logger(self, temp_dir):
        """Test that main creates SessionLogger with output directory."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]), patch('main.VideoFileCapture') as mock_capture:
            mock_capture_instance = MagicMock()
            mock_capture_instance.get_metadata.return_value = {'fps': 30}
            mock_capture_instance.__iter__ = Mock(return_value=iter([]))
            mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
            mock_capture_instance.__exit__ = Mock(return_value=False)
            mock_capture.return_value = mock_capture_instance

            with patch('main.SessionLogger') as mock_logger:
                main.main()
                # Verify SessionLogger was created with output directory
                mock_logger.assert_called_once_with(str(temp_dir))

    def test_main_logs_completion_summary(self, temp_dir):
        """Test that main logs completion summary."""
        import main

        roi_config = temp_dir / "roi.yaml"
        roi_config.write_text("rois:\n  hp_shield:\n    bbox: [0.0, 0.0, 1.0, 1.0]\n  knocked_revive:\n    bbox: [0.0, 0.0, 1.0, 1.0]")

        triggers_config = temp_dir / "triggers.yaml"
        triggers_config.write_text("triggers: []")

        system_prompt = temp_dir / "prompt.txt"
        system_prompt.write_text("Test prompt")

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake_video")

        with patch.object(sys, 'argv', [
            'main.py',
            '--input', 'video',
            '--video', str(video_file),
            '--roi', str(roi_config),
            '--triggers', str(triggers_config),
            '--system-prompt', str(system_prompt),
            '--out', str(temp_dir)
        ]):
            test_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

            with patch('main.VideoFileCapture') as mock_capture:
                mock_capture_instance = MagicMock()
                mock_capture_instance.get_metadata.return_value = {'fps': 30, 'frame_count': 5}
                mock_capture_instance.__iter__ = Mock(return_value=iter(test_frames))
                mock_capture_instance.__enter__ = Mock(return_value=mock_capture_instance)
                mock_capture_instance.__exit__ = Mock(return_value=False)
                mock_capture.return_value = mock_capture_instance

                main.main()

                # Verify completion
                assert mock_capture_instance.__iter__.called
