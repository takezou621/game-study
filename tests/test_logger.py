"""Tests for SessionLogger."""

import json
from pathlib import Path

from utils.logger import SessionLogger


class TestSessionLogger:
    """Test SessionLogger functionality."""

    def test_init(self, temp_session_dir):
        """Test SessionLogger initialization."""
        logger = SessionLogger(temp_session_dir)

        assert logger.session_dir == Path(temp_session_dir)
        assert logger.state_log_path == Path(temp_session_dir) / "state.jsonl"
        assert logger.trigger_log_path == Path(temp_session_dir) / "triggers.jsonl"
        assert logger.response_log_path == Path(temp_session_dir) / "responses.jsonl"
        assert logger.error_log_path == Path(temp_session_dir) / "errors.log"

    def test_init_creates_directory(self, tmp_path):
        """Test that init creates the session directory."""
        session_dir = tmp_path / "new_session_dir"
        assert not session_dir.exists()

        logger = SessionLogger(str(session_dir))
        assert session_dir.exists()
        assert session_dir.is_dir()

    def test_log_state(self, temp_session_dir):
        """Test logging game state."""
        logger = SessionLogger(temp_session_dir)

        test_state = {
            "player": {"status": {"hp": {"value": 75}}},
            "timestamp": 12345,
        }

        logger.log_state(test_state)

        # Verify file was created and contains correct data
        assert logger.state_log_path.exists()

        with open(logger.state_log_path) as f:
            content = f.read()
            assert content.strip() == json.dumps(test_state, ensure_ascii=False)

    def test_log_state_multiple(self, temp_session_dir):
        """Test logging multiple states."""
        logger = SessionLogger(temp_session_dir)

        states = [
            {"player": {"status": {"hp": {"value": 100}}}},
            {"player": {"status": {"value": 75}}},
            {"player": {"status": {"hp": {"value": 50}}}},
        ]

        for state in states:
            logger.log_state(state)

        # Verify all states were logged
        with open(logger.state_log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0]) == states[0]
        assert json.loads(lines[1]) == states[1]
        assert json.loads(lines[2]) == states[2]

    def test_log_trigger(self, temp_session_dir):
        """Test logging trigger event."""
        logger = SessionLogger(temp_session_dir)

        test_trigger = {
            "rule_id": "test_rule",
            "rule_name": "Test Rule",
            "priority": 0,
            "timestamp": 12345,
        }

        logger.log_trigger(test_trigger)

        # Verify file was created and contains correct data
        assert logger.trigger_log_path.exists()

        with open(logger.trigger_log_path) as f:
            content = f.read()
            assert content.strip() == json.dumps(test_trigger, ensure_ascii=False)

    def test_log_trigger_multiple(self, temp_session_dir):
        """Test logging multiple triggers."""
        logger = SessionLogger(temp_session_dir)

        triggers = [
            {"rule_id": "rule1", "timestamp": 1000},
            {"rule_id": "rule2", "timestamp": 2000},
            {"rule_id": "rule3", "timestamp": 3000},
        ]

        for trigger in triggers:
            logger.log_trigger(trigger)

        # Verify all triggers were logged
        with open(logger.trigger_log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_log_response(self, temp_session_dir):
        """Test logging AI response."""
        logger = SessionLogger(temp_session_dir)

        test_response = {
            "trigger_id": "test_trigger",
            "response_text": "Test response",
            "timestamp": 12345,
        }

        logger.log_response(test_response)

        # Verify file was created and contains correct data
        assert logger.response_log_path.exists()

        with open(logger.response_log_path) as f:
            content = f.read()
            assert content.strip() == json.dumps(test_response, ensure_ascii=False)

    def test_log_response_multiple(self, temp_session_dir):
        """Test logging multiple responses."""
        logger = SessionLogger(temp_session_dir)

        responses = [
            {"response": "Response 1"},
            {"response": "Response 2"},
            {"response": "Response 3"},
        ]

        for response in responses:
            logger.log_response(response)

        # Verify all responses were logged
        with open(logger.response_log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_log_error_without_exception(self, temp_session_dir):
        """Test logging error message without exception."""
        logger = SessionLogger(temp_session_dir)

        logger.log_error("Test error message")

        # Verify file was created
        assert logger.error_log_path.exists()

        with open(logger.error_log_path) as f:
            content = f.read()

        assert "Test error message" in content
        assert "Exception:" not in content

    def test_log_error_with_exception(self, temp_session_dir):
        """Test logging error message with exception."""
        logger = SessionLogger(temp_session_dir)

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_error("Something went wrong", e)

        # Verify file was created
        assert logger.error_log_path.exists()

        with open(logger.error_log_path) as f:
            content = f.read()

        assert "Something went wrong" in content
        assert "ValueError" in content
        assert "Test exception" in content

    def test_log_error_multiple(self, temp_session_dir):
        """Test logging multiple errors."""
        logger = SessionLogger(temp_session_dir)

        logger.log_error("Error 1")
        logger.log_error("Error 2")
        logger.log_error("Error 3")

        # Verify all errors were logged
        with open(logger.error_log_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_log_state_append_mode(self, temp_session_dir):
        """Test that logging appends to existing file."""
        logger = SessionLogger(temp_session_dir)

        logger.log_state({"state": 1})

        # Create a new logger instance (same directory)
        logger2 = SessionLogger(temp_session_dir)
        logger2.log_state({"state": 2})

        # Both states should be in the file
        with open(logger.state_log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_get_session_info(self, temp_session_dir):
        """Test getting session information."""
        logger = SessionLogger(temp_session_dir)

        info = logger.get_session_info()

        assert "session_dir" in info
        assert "state_log_path" in info
        assert "trigger_log_path" in info
        assert "response_log_path" in info
        assert "created_at" in info

        assert info["session_dir"] == str(temp_session_dir)
        assert info["state_log_path"] == str(logger.state_log_path)
        assert info["trigger_log_path"] == str(logger.trigger_log_path)
        assert info["response_log_path"] == str(logger.response_log_path)

    def test_jsonl_format(self, temp_session_dir):
        """Test that JSONL format is correct (one JSON per line)."""
        logger = SessionLogger(temp_session_dir)

        logger.log_state({"seq": 1})
        logger.log_state({"seq": 2})
        logger.log_state({"seq": 3})

        with open(logger.state_log_path) as f:
            for i, line in enumerate(f, start=1):
                data = json.loads(line)
                assert data["seq"] == i

    def test_unicode_handling(self, temp_session_dir):
        """Test that Unicode characters are handled correctly."""
        logger = SessionLogger(temp_session_dir)

        test_data = {
            "message": "日本語テスト 🎮",
            "emoji": "🔥💯⭐",
        }

        logger.log_state(test_data)

        with open(logger.state_log_path, encoding="utf-8") as f:
            content = f.read()

        assert "日本語テスト" in content
        assert "🎮" in content

    def test_complex_nested_state(self, temp_session_dir):
        """Test logging complex nested state."""
        logger = SessionLogger(temp_session_dir)

        complex_state = {
            "player": {
                "status": {
                    "hp": {"value": 75, "source": "roi", "confidence": 0.9},
                    "shield": {"value": 50, "source": "roi", "confidence": 0.95},
                },
                "weapon": {
                    "name": {"value": "Assault Rifle", "source": "yolo"},
                    "ammo": {"value": 30, "source": "ocr"},
                },
            },
            "world": {
                "storm": {
                    "in_storm": {"value": False},
                    "damage": {"value": 5.0},
                }
            },
        }

        logger.log_state(complex_state)

        with open(logger.state_log_path) as f:
            loaded = json.loads(f.read())

        assert loaded == complex_state

    def test_empty_state(self, temp_session_dir):
        """Test logging empty state."""
        logger = SessionLogger(temp_session_dir)

        logger.log_state({})

        with open(logger.state_log_path) as f:
            loaded = json.loads(f.read())

        assert loaded == {}

    def test_log_to_different_files_independently(self, temp_session_dir):
        """Test that different log files work independently."""
        logger = SessionLogger(temp_session_dir)

        logger.log_state({"type": "state"})
        logger.log_trigger({"type": "trigger"})
        logger.log_response({"type": "response"})
        logger.log_error("Error message")

        # Verify each file has only its content
        with open(logger.state_log_path) as f:
            assert json.loads(f.read())["type"] == "state"

        with open(logger.trigger_log_path) as f:
            assert json.loads(f.read())["type"] == "trigger"

        with open(logger.response_log_path) as f:
            assert json.loads(f.read())["type"] == "response"

        with open(logger.error_log_path) as f:
            assert "Error message" in f.read()
