"""Integration tests for game-study."""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from trigger.engine import TriggerEngine
from vision.state_builder import StateBuilder
from dialogue.templates import DialogueTemplateManager
from utils.logger import SessionLogger


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system."""

    def test_state_builder_to_trigger_engine_workflow(self, sample_triggers_config_path):
        """Test workflow from StateBuilder to TriggerEngine."""
        # Create state builder and build state
        builder = StateBuilder()
        builder.update_hp(25, "roi", 0.95)
        builder.update_shield(0, "roi", 1.0)
        builder.update_in_storm(False, "roi", 1.0)

        state = builder.get_state()

        # Create trigger engine and evaluate
        engine = TriggerEngine(sample_triggers_config_path)
        result = engine.evaluate_triggers(state, "combat")

        # Should trigger low HP warning
        assert result is not None
        assert result["rule_id"] == "test_low_hp"
        assert result["template"] == "Low HP in combat!"

    def test_full_pipeline_state_to_template(self, sample_triggers_config_path):
        """Test full pipeline: state -> trigger -> template."""
        # Setup
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)
        template_manager = DialogueTemplateManager()

        # Add template for the test trigger
        template_manager.add_template("test_low_hp_combat", "Low HP in combat!")
        template_manager.add_template("test_low_hp_non_combat", "Low HP not in combat.")

        # Build critical state
        builder.update_hp(20, "roi", 0.95)
        state = builder.get_state()

        # Get trigger result
        trigger_result = engine.evaluate_triggers(state, "combat")
        assert trigger_result is not None

        # Get template
        template = template_manager.get_template(
            trigger_result["rule_id"],
            trigger_result["movement_state"]
        )

        assert template is not None
        assert "Low HP" in template or "critical" in template.lower()

    def test_state_builder_logging_workflow(self, temp_session_dir):
        """Test workflow from StateBuilder to logging."""
        builder = StateBuilder()
        logger = SessionLogger(temp_session_dir)

        # Build and update state
        builder.update_hp(75, "test", 0.9)
        builder.update_weapon_name("Assault Rifle", "yolo", 0.85)
        state = builder.get_state()

        # Log state
        logger.log_state(state)

        # Verify logged
        assert logger.state_log_path.exists()
        with open(logger.state_log_path, "r") as f:
            logged_state = json.loads(f.read().strip())

        assert logged_state["player"]["status"]["hp"]["value"] == 75
        assert logged_state["player"]["weapon"]["name"]["value"] == "Assault Rifle"

    def test_trigger_to_logging_workflow(self, sample_triggers_config_path, temp_session_dir):
        """Test workflow from TriggerEngine to logging."""
        engine = TriggerEngine(sample_triggers_config_path)
        logger = SessionLogger(temp_session_dir)

        # Create state that triggers
        state = {
            "player": {"status": {"hp": {"value": 25}}},
            "world": {"storm": {"in_storm": {"value": False}}},
            "session": {"inactivity_duration_ms": {"value": 0}},
        }

        # Evaluate and log
        trigger_result = engine.evaluate_triggers(state, "combat")
        if trigger_result:
            logger.log_trigger(trigger_result)

        # Verify logged
        assert logger.trigger_log_path.exists()

    def test_movement_state_affects_triggers(self, sample_triggers_config_path):
        """Test that movement state affects trigger selection."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)

        # Set low HP
        builder.update_hp(25, "test", 0.9)
        state = builder.get_state()

        # Test combat state
        result_combat = engine.evaluate_triggers(state, "combat")
        assert result_combat is not None
        assert result_combat["movement_state"] == "combat"

        # Reset cooldown
        for rule in engine.rules:
            rule.last_triggered_ms = 0

        # Test non-combat state
        result_non_combat = engine.evaluate_triggers(state, "non_combat")
        assert result_non_combat is not None
        assert result_non_combat["movement_state"] == "non_combat"

    def test_multiple_state_updates(self, sample_triggers_config_path):
        """Test multiple state updates and trigger evaluations."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)

        # Initial state - no trigger
        builder.update_hp(100, "test", 1.0)
        state1 = builder.get_state()
        result1 = engine.evaluate_triggers(state1, "combat")
        assert result1 is None

        # HP drops - trigger fires
        builder.update_hp(25, "test", 0.9)
        state2 = builder.get_state()
        result2 = engine.evaluate_triggers(state2, "combat")
        assert result2 is not None

        # HP recovers but still on cooldown - no trigger
        builder.update_hp(80, "test", 1.0)
        state3 = builder.get_state()
        result3 = engine.evaluate_triggers(state3, "combat")
        assert result3 is None  # On cooldown

    def test_storm_detection_workflow(self, sample_triggers_config_path):
        """Test storm detection and trigger workflow."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)

        # Player in storm
        builder.update_in_storm(True, "roi", 1.0)
        builder.update_storm_damage(5.0, "roi", 1.0)
        state = builder.get_state()

        result = engine.evaluate_triggers(state, "non_combat")

        assert result is not None
        assert result["rule_id"] == "test_storm"
        # Template should indicate storm-related action
        assert "safe zone" in result["template"].lower() or "storm" in result["template"].lower()

    def test_state_builder_reset(self, sample_triggers_config_path):
        """Test state builder reset and re-evaluation."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)

        # Set low HP
        builder.update_hp(25, "test", 0.9)
        state1 = builder.get_state()
        result1 = engine.evaluate_triggers(state1, "combat")
        assert result1 is not None

        # Reset state
        builder.reset()
        state2 = builder.get_state()

        # Should be back to defaults
        assert state2["player"]["status"]["hp"]["value"] == 100

    def test_priority_ordering_integration(self, sample_triggers_config_path):
        """Test that higher priority triggers fire first."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)

        # Create state that matches both low_hp (P0) and storm (P1)
        builder.update_hp(25, "test", 0.9)
        builder.update_in_storm(True, "test", 1.0)
        state = builder.get_state()

        result = engine.evaluate_triggers(state, "combat")

        # P0 should fire (low_hp has priority 0, storm has priority 1)
        assert result is not None
        assert result["priority"] == 0
        assert result["rule_id"] == "test_low_hp"

    def test_template_variable_substitution_integration(self, sample_triggers_config_path):
        """Test template variable substitution in full workflow."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)
        template_manager = DialogueTemplateManager()

        # Setup state with new weapon
        builder.update_weapon_name("Shotgun", "yolo", 0.9)
        builder.update_hp(100, "test", 1.0)  # Not in danger

        # For this test, we'll manually test template substitution
        template = template_manager.get_template(
            "p2_weapon",
            "non_combat",
            weapon_name="Shotgun",
            situation="close range combat"
        )

        assert template is not None
        assert "Shotgun" in template
        assert "close range combat" in template

    def test_combat_suppression_integration(self, sample_triggers_config_path):
        """Test that lower priority triggers are suppressed during combat."""
        # Create custom config with P2 rule
        import yaml
        import tempfile

        config_data = {
            "triggers": [
                {
                    "id": "p0_critical",
                    "name": "P0 Critical",
                    "priority": 0,
                    "enabled": True,
                    "conditions": [{"field": "player.status.hp", "operator": "lt", "value": 20}],
                    "template": {"combat": "Critical!", "non_combat": "Critical low HP"},
                    "cooldown_ms": 0,
                },
                {
                    "id": "p2_chatter",
                    "name": "P2 Chatter",
                    "priority": 2,
                    "enabled": True,
                    "conditions": [{"field": "session.inactivity_duration_ms", "operator": "gt", "value": 1000}],
                    "template": {"combat": None, "non_combat": "Chat message"},
                    "cooldown_ms": 0,
                },
            ],
            "settings": {
                "cooldown_enabled": True,
                "combat_suppress_priority": [2, 3],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            builder = StateBuilder()
            engine = TriggerEngine(config_path)

            # Low HP and long inactivity (both triggers match)
            builder.update_hp(15, "test", 0.9)

            state = builder.get_state()
            state["session"]["inactivity_duration_ms"]["value"] = 5000

            # In combat, P2 should be suppressed
            result = engine.evaluate_triggers(state, "combat")
            assert result is not None
            assert result["priority"] == 0

        finally:
            Path(config_path).unlink()

    def test_cooldown_persistence_integration(self, sample_triggers_config_path):
        """Test that cooldown persists across evaluations."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)

        # Set low HP
        builder.update_hp(25, "test", 0.9)
        state = builder.get_state()

        # First trigger
        result1 = engine.evaluate_triggers(state, "combat")
        assert result1 is not None

        # Immediate second - should be on cooldown
        result2 = engine.evaluate_triggers(state, "combat")
        assert result2 is None

        # Get the rule and check it was updated
        rule = engine.get_rule_by_id(result1["rule_id"])
        assert rule.last_triggered_ms > 0

    def test_complete_session_workflow(self, sample_triggers_config_path, temp_session_dir):
        """Test complete workflow: state -> trigger -> template -> logging."""
        builder = StateBuilder()
        engine = TriggerEngine(sample_triggers_config_path)
        template_manager = DialogueTemplateManager()
        logger = SessionLogger(temp_session_dir)

        # Add templates for test triggers
        template_manager.add_template("test_low_hp_combat", "Low HP in combat!")
        template_manager.add_template("test_low_hp_non_combat", "Low HP not in combat.")

        # 1. Build state
        builder.update_hp(20, "roi", 0.95)
        builder.update_shield(0, "roi", 1.0)
        state = builder.get_state()

        # 2. Log state
        logger.log_state(state)

        # 3. Evaluate triggers
        trigger_result = engine.evaluate_triggers(state, "combat")
        assert trigger_result is not None

        # 4. Log trigger
        logger.log_trigger(trigger_result)

        # 5. Get template
        template = template_manager.get_template(
            trigger_result["rule_id"],
            trigger_result["movement_state"]
        )
        assert template is not None

        # 6. Log response
        response = {
            "trigger_id": trigger_result["rule_id"],
            "template": template,
            "movement_state": trigger_result["movement_state"],
        }
        logger.log_response(response)

        # 7. Verify all logs
        assert logger.state_log_path.exists()
        assert logger.trigger_log_path.exists()
        assert logger.response_log_path.exists()

    def test_error_handling_integration(self, sample_triggers_config_path, temp_session_dir):
        """Test error handling in integration workflow."""
        logger = SessionLogger(temp_session_dir)

        # Simulate an error
        try:
            raise ValueError("Simulated integration error")
        except ValueError as e:
            logger.log_error("Integration test error", e)

        # Verify error was logged
        assert logger.error_log_path.exists()
        with open(logger.error_log_path, "r") as f:
            content = f.read()
            assert "Integration test error" in content
            assert "ValueError" in content

    def test_multiple_rules_same_priority(self, tmp_path):
        """Test handling of multiple rules with same priority."""
        import yaml

        config_data = {
            "triggers": [
                {
                    "id": "rule_a",
                    "name": "Rule A",
                    "priority": 0,
                    "enabled": True,
                    "conditions": [{"field": "player.status.hp", "operator": "lt", "value": 50}],
                    "template": {"combat": "A", "non_combat": "A"},
                    "cooldown_ms": 0,
                },
                {
                    "id": "rule_b",
                    "name": "Rule B",
                    "priority": 0,
                    "enabled": True,
                    "conditions": [{"field": "player.status.shield", "operator": "eq", "value": 0}],
                    "template": {"combat": "B", "non_combat": "B"},
                    "cooldown_ms": 0,
                },
            ],
            "settings": {"cooldown_enabled": True},
        }

        config_path = tmp_path / "triggers.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        engine = TriggerEngine(str(config_path))

        # State that matches both rules
        state = {
            "player": {
                "status": {"hp": {"value": 25}, "shield": {"value": 0}}
            },
            "session": {"inactivity_duration_ms": {"value": 0}},
            "world": {"storm": {"in_storm": {"value": False}}},
        }
        result = engine.evaluate_triggers(state, "combat")

        # Should fire one of them (first in list after sorting by priority)
        assert result is not None
        assert result["rule_id"] in ["rule_a", "rule_b"]

    def test_state_movement_state_detection(self):
        """Test StateBuilder movement state detection."""
        builder = StateBuilder()

        # Non-combat: high HP, not in storm
        builder.update_hp(100, "test", 1.0)
        builder.update_in_storm(False, "test", 1.0)
        assert builder.get_movement_state() == "non_combat"

        # Combat: low HP
        builder.update_hp(40, "test", 1.0)
        assert builder.get_movement_state() == "combat"

        # Combat: in storm
        builder.update_hp(100, "test", 1.0)
        builder.update_in_storm(True, "test", 1.0)
        assert builder.get_movement_state() == "combat"

    def test_template_short_form_integration(self):
        """Test short template form in integration context."""
        template_manager = DialogueTemplateManager()

        # Create a long template (need full key with movement state)
        template_manager.add_template(
            "long_test_combat",
            "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        )

        short = template_manager.get_short_template("long_test", "combat")

        assert short is not None
        sentences = short.split(". ")
        assert len(sentences) <= 2
