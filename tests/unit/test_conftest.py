"""Test that conftest fixtures work correctly."""

import pytest


class TestFixtures:
    """Test that fixtures are properly loaded."""

    def test_base_state(self, base_state):
        """Test base_state fixture."""
        assert "player" in base_state
        assert base_state["player"]["status"]["hp"]["value"] == 100

    def test_low_hp_state(self, low_hp_state):
        """Test low_hp_state fixture."""
        assert low_hp_state["player"]["status"]["hp"]["value"] == 25

    def test_in_storm_state(self, in_storm_state):
        """Test in_storm_state fixture."""
        assert in_storm_state["world"]["storm"]["in_storm"]["value"] is True

    def test_knocked_state(self, knocked_state):
        """Test knocked_state fixture."""
        assert knocked_state["player"]["status"]["is_knocked"]["value"] is True
        assert knocked_state["player"]["status"]["hp"]["value"] == 0

    def test_temp_dir(self, temp_dir):
        """Test temp_dir fixture."""
        assert temp_dir.exists()
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello")
        assert test_file.read_text() == "hello"

    def test_state_builder(self, state_builder):
        """Test state_builder fixture."""
        state = state_builder.get_state()
        assert "player" in state
        assert "world" in state

    def test_trigger_condition(self, trigger_condition):
        """Test trigger_condition fixture."""
        assert trigger_condition.field == "player.status.hp"
        assert trigger_condition.operator == "lt"
        assert trigger_condition.value == 30

    def test_trigger_rule(self, trigger_rule):
        """Test trigger_rule fixture."""
        assert trigger_rule.id == "test_low_hp"
        assert trigger_rule.priority == 0
        assert trigger_rule.enabled is True

    def test_webrtc_server(self, webrtc_server):
        """Test webrtc_server fixture."""
        assert webrtc_server.host == "127.0.0.1"
        assert webrtc_server.port == 8080
