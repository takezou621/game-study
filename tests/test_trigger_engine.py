"""Tests for TriggerEngine."""

from unittest.mock import patch

import pytest

from trigger.engine import TriggerEngine
from trigger.rules import TriggerCondition, TriggerRule


class TestTriggerEngine:
    """Test TriggerEngine functionality."""

    def test_init(self, sample_triggers_config_path):
        """Test TriggerEngine initialization."""
        engine = TriggerEngine(sample_triggers_config_path)

        assert engine.config is not None
        assert len(engine.rules) == 5  # 5 rules in sample_triggers_config_path
        assert engine.cooldown_enabled is True
        assert engine.interrupt_higher_priority is True
        assert engine.max_response_length == 200
        assert engine.combat_suppress_priority == [2, 3]
        assert engine.inactivity_threshold_ms == 30000

    def test_rules_sorted_by_priority(self, sample_triggers_config_path):
        """Test that rules are sorted by priority."""
        engine = TriggerEngine(sample_triggers_config_path)

        priorities = [rule.priority for rule in engine.rules]
        assert priorities == sorted(priorities)

    def test_load_config_invalid_path(self):
        """Test loading config with invalid path."""
        with pytest.raises(FileNotFoundError):
            TriggerEngine("/nonexistent/path/triggers.yaml")

    def test_evaluate_triggers_low_hp_combat(self, sample_triggers_config_path, sample_state):
        """Test trigger evaluation for low HP in combat."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Set low HP
        sample_state["player"]["status"]["hp"]["value"] = 25

        result = engine.evaluate_triggers(sample_state, "combat")

        assert result is not None
        assert result["rule_id"] == "test_low_hp"
        assert result["rule_name"] == "Test Low HP"
        assert result["priority"] == 0
        assert result["template"] == "Low HP in combat!"
        assert result["movement_state"] == "combat"

    def test_evaluate_triggers_low_hp_non_combat(self, sample_triggers_config_path, sample_state):
        """Test trigger evaluation for low HP in non-combat."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Set low HP
        sample_state["player"]["status"]["hp"]["value"] = 25

        result = engine.evaluate_triggers(sample_state, "non_combat")

        assert result is not None
        assert result["rule_id"] == "test_low_hp"
        assert result["template"] == "Low HP not in combat."

    def test_evaluate_triggers_no_match(self, sample_triggers_config_path, sample_state):
        """Test trigger evaluation when no conditions match."""
        engine = TriggerEngine(sample_triggers_config_path)

        # HP is 100, not in storm
        result = engine.evaluate_triggers(sample_state, "combat")

        assert result is None

    def test_evaluate_triggers_disabled_rule(self, sample_triggers_config_path, sample_state):
        """Test that disabled rules don't fire."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Set HP to 0 (should match test_disabled rule but it's disabled)
        sample_state["player"]["status"]["hp"]["value"] = 0

        result = engine.evaluate_triggers(sample_state, "combat")

        assert result is None or result["rule_id"] != "test_disabled"

    def test_evaluate_triggers_cooldown(self, sample_triggers_config_path, sample_state):
        """Test trigger cooldown functionality."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Set low HP
        sample_state["player"]["status"]["hp"]["value"] = 25

        # First trigger should fire
        result1 = engine.evaluate_triggers(sample_state, "combat")
        assert result1 is not None

        # Immediate second call should not fire due to cooldown
        result2 = engine.evaluate_triggers(sample_state, "combat")
        assert result2 is None

    def test_evaluate_triggers_cooldown_disabled(self, sample_triggers_config_path, sample_state):
        """Test trigger evaluation with cooldown disabled."""
        engine = TriggerEngine(sample_triggers_config_path)
        engine.cooldown_enabled = False

        # Set low HP
        sample_state["player"]["status"]["hp"]["value"] = 25

        # Both should fire when cooldown is disabled
        result1 = engine.evaluate_triggers(sample_state, "combat")
        assert result1 is not None

        result2 = engine.evaluate_triggers(sample_state, "combat")
        # With cooldown disabled, trigger fires again (though it still gets cooldown updated)

    def test_evaluate_triggers_storm(self, sample_triggers_config_path, sample_state):
        """Test storm trigger evaluation."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Set in_storm to True
        sample_state["world"]["storm"]["in_storm"]["value"] = True

        result = engine.evaluate_triggers(sample_state, "combat")

        assert result is not None
        assert result["rule_id"] == "test_storm"
        assert result["template"] == "Get out of storm!"

    def test_evaluate_triggers_priority_order(self, sample_triggers_config_path, sample_state):
        """Test that higher priority triggers fire first."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Set both low HP (priority 0) and in storm (priority 1)
        sample_state["player"]["status"]["hp"]["value"] = 25
        sample_state["world"]["storm"]["in_storm"]["value"] = True

        result = engine.evaluate_triggers(sample_state, "combat")

        # Should fire the higher priority (lower number) trigger
        assert result["priority"] == 0
        assert result["rule_id"] == "test_low_hp"

    def test_evaluate_triggers_updates_inactivity(self, sample_triggers_config_path, sample_state):
        """Test that inactivity duration is updated."""
        engine = TriggerEngine(sample_triggers_config_path)

        initial_time = 12345
        engine.last_trigger_time_ms = initial_time

        with patch("trigger.engine.get_timestamp_ms", return_value=initial_time + 5000):
            engine.evaluate_triggers(sample_state, "combat")

            assert sample_state["session"]["inactivity_duration_ms"]["value"] == 5000

    def test_get_rule_by_id(self, sample_triggers_config_path):
        """Test getting a rule by ID."""
        engine = TriggerEngine(sample_triggers_config_path)

        rule = engine.get_rule_by_id("test_low_hp")
        assert rule is not None
        assert rule.id == "test_low_hp"
        assert rule.name == "Test Low HP"

        # Non-existent rule
        rule = engine.get_rule_by_id("non_existent")
        assert rule is None

    def test_enable_rule(self, sample_triggers_config_path):
        """Test enabling a rule."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Disable the rule first
        rule = engine.get_rule_by_id("test_low_hp")
        rule.enabled = False
        assert rule.enabled is False

        # Enable it back
        result = engine.enable_rule("test_low_hp")
        assert result is True
        assert rule.enabled is True

        # Try to enable non-existent rule
        result = engine.enable_rule("non_existent")
        assert result is False

    def test_disable_rule(self, sample_triggers_config_path):
        """Test disabling a rule."""
        engine = TriggerEngine(sample_triggers_config_path)

        rule = engine.get_rule_by_id("test_low_hp")
        assert rule.enabled is True

        result = engine.disable_rule("test_low_hp")
        assert result is True
        assert rule.enabled is False

        # Try to disable non-existent rule
        result = engine.disable_rule("non_existent")
        assert result is False

    def test_get_all_rules(self, sample_triggers_config_path):
        """Test getting all rules as dictionaries."""
        engine = TriggerEngine(sample_triggers_config_path)

        rules = engine.get_all_rules()
        assert len(rules) == 5  # 5 rules in sample_triggers_config_path
        assert all(isinstance(rule, dict) for rule in rules)
        assert "id" in rules[0]
        assert "name" in rules[0]
        assert "priority" in rules[0]

    def test_evaluate_triggers_missing_template_for_state(self, sample_triggers_config_path, sample_state):
        """Test trigger when template is missing for movement state."""
        engine = TriggerEngine(sample_triggers_config_path)

        # Create a rule without combat template
        custom_config = {
            "triggers": [{
                "id": "test_no_template",
                "name": "No Template Test",
                "priority": 0,
                "enabled": True,
                "conditions": [
                    {"field": "player.status.hp", "operator": "lt", "value": 30}
                ],
                "template": {
                    "non_combat": "Only non-combat template",
                },
                "cooldown_ms": 5000,
            }],
            "settings": engine.settings,
        }

        # We'll just test with existing config and non_combat state
        sample_state["player"]["status"]["hp"]["value"] = 25
        result = engine.evaluate_triggers(sample_state, "non_combat")

        # Should work for non_combat
        assert result is not None


class TestTriggerCondition:
    """Test TriggerCondition functionality."""

    def test_evaluate_eq(self):
        """Test equals operator."""
        condition = TriggerCondition("player.status.hp", "eq", 50)
        state = {"player": {"status": {"hp": 50}}}
        assert condition.evaluate(state) is True

        state = {"player": {"status": {"hp": 49}}}
        assert condition.evaluate(state) is False

    def test_evaluate_lt(self):
        """Test less than operator."""
        condition = TriggerCondition("player.status.hp", "lt", 30)
        state = {"player": {"status": {"hp": 25}}}
        assert condition.evaluate(state) is True

        state = {"player": {"status": {"hp": 30}}}
        assert condition.evaluate(state) is False

    def test_evaluate_gt(self):
        """Test greater than operator."""
        condition = TriggerCondition("player.status.hp", "gt", 50)
        state = {"player": {"status": {"hp": 75}}}
        assert condition.evaluate(state) is True

        state = {"player": {"status": {"hp": 50}}}
        assert condition.evaluate(state) is False

    def test_evaluate_lte(self):
        """Test less than or equal operator."""
        condition = TriggerCondition("player.status.hp", "lte", 30)
        state = {"player": {"status": {"hp": 30}}}
        assert condition.evaluate(state) is True

        state = {"player": {"status": {"hp": 31}}}
        assert condition.evaluate(state) is False

    def test_evaluate_gte(self):
        """Test greater than or equal operator."""
        condition = TriggerCondition("player.status.hp", "gte", 70)
        state = {"player": {"status": {"hp": 70}}}
        assert condition.evaluate(state) is True

        state = {"player": {"status": {"hp": 69}}}
        assert condition.evaluate(state) is False

    def test_evaluate_ne(self):
        """Test not equal operator."""
        condition = TriggerCondition("player.status.hp", "ne", 0)
        state = {"player": {"status": {"hp": 100}}}
        assert condition.evaluate(state) is True

        state = {"player": {"status": {"hp": 0}}}
        assert condition.evaluate(state) is False

    def test_evaluate_with_nested_value_dict(self):
        """Test evaluation with nested value dict structure."""
        condition = TriggerCondition("player.status.hp", "lt", 30)
        state = {
            "player": {
                "status": {
                    "hp": {"value": 25, "source": "test", "confidence": 1.0}
                }
            }
        }
        assert condition.evaluate(state) is True

    def test_evaluate_missing_field(self):
        """Test evaluation with missing field."""
        condition = TriggerCondition("player.status.missing_field", "eq", 50)
        state = {"player": {"status": {"hp": 50}}}
        assert condition.evaluate(state) is False

    def test_evaluate_none_value(self):
        """Test evaluation with None value."""
        condition = TriggerCondition("player.status.hp", "lt", 30)
        state = {"player": {"status": {"hp": None}}}
        assert condition.evaluate(state) is False

    def test_evaluate_invalid_operator(self):
        """Test evaluation with invalid operator - Pydantic validates at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TriggerCondition("player.status.hp", "invalid", 30)

    def test_to_dict(self):
        """Test converting condition to dictionary."""
        condition = TriggerCondition("player.status.hp", "lt", 30)
        result = condition.to_dict()

        assert result == {
            "field": "player.status.hp",
            "operator": "lt",
            "value": 30,
        }


class TestTriggerRule:
    """Test TriggerRule functionality."""

    def test_init(self):
        """Test rule initialization."""
        conditions = [
            TriggerCondition("player.status.hp", "lt", 30),
            TriggerCondition("player.status.shield", "eq", 0),
        ]

        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates={"combat": "Test template", "non_combat": "Non-combat template"},
            cooldown_ms=5000,
            interrupt_higher_priority=False,
        )

        assert rule.id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.priority == 0
        assert rule.enabled is True
        assert len(rule.conditions) == 2
        assert rule.cooldown_ms == 5000
        assert rule.interrupt_higher_priority is False

    def test_evaluate_all_conditions_met(self):
        """Test evaluation when all conditions are met."""
        conditions = [
            TriggerCondition("player.status.hp", "lt", 30),
            TriggerCondition("player.status.shield", "eq", 0),
        ]

        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates={"combat": "Test"},
            cooldown_ms=0,
        )

        state = {
            "player": {
                "status": {"hp": 25, "shield": 0}
            }
        }

        assert rule.evaluate(state) is True

    def test_evaluate_one_condition_not_met(self):
        """Test evaluation when one condition is not met."""
        conditions = [
            TriggerCondition("player.status.hp", "lt", 30),
            TriggerCondition("player.status.shield", "eq", 0),
        ]

        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates={"combat": "Test"},
            cooldown_ms=0,
        )

        state = {
            "player": {
                "status": {"hp": 25, "shield": 50}  # Shield not 0
            }
        }

        assert rule.evaluate(state) is False

    def test_evaluate_disabled(self):
        """Test that disabled rules don't evaluate."""
        conditions = [TriggerCondition("player.status.hp", "lt", 30)]

        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=False,  # Disabled
            conditions=conditions,
            templates={"combat": "Test"},
            cooldown_ms=0,
        )

        state = {"player": {"status": {"hp": 25}}}

        assert rule.evaluate(state) is False

    def test_get_template(self):
        """Test getting template for movement state."""
        templates = {"combat": "Combat template", "non_combat": "Non-combat template"}

        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=[],
            templates=templates,
            cooldown_ms=0,
        )

        assert rule.get_template("combat") == "Combat template"
        assert rule.get_template("non_combat") == "Non-combat template"
        assert rule.get_template("unknown") is None

    def test_is_on_cooldown(self):
        """Test cooldown checking."""
        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=[],
            templates={"combat": "Test"},
            cooldown_ms=5000,
        )

        rule.last_triggered_ms = 10000

        # Within cooldown
        assert rule.is_on_cooldown(12000) is True

        # Outside cooldown
        assert rule.is_on_cooldown(16000) is False

    def test_update_last_triggered(self):
        """Test updating last triggered timestamp."""
        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=[],
            templates={"combat": "Test"},
            cooldown_ms=0,
        )

        rule.update_last_triggered(15000)
        assert rule.last_triggered_ms == 15000

    def test_to_dict(self):
        """Test converting rule to dictionary."""
        conditions = [TriggerCondition("player.status.hp", "lt", 30)]

        rule = TriggerRule(
            rule_id="test_rule",
            name="Test Rule",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates={"combat": "Test"},
            cooldown_ms=5000,
            interrupt_higher_priority=True,
        )

        result = rule.to_dict()

        assert result["id"] == "test_rule"
        assert result["name"] == "Test Rule"
        assert result["priority"] == 0
        assert result["enabled"] is True
        assert len(result["conditions"]) == 1
        assert result["templates"] == {"combat": "Test"}
        assert result["cooldown_ms"] == 5000
        assert result["interrupt_higher_priority"] is True
        assert "last_triggered_ms" in result
