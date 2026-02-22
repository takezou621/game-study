"""Tests for TriggerCondition and TriggerRule classes."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

# Direct module import to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"
RULES_PATH = SRC_PATH / "trigger" / "rules.py"

spec = importlib.util.spec_from_file_location("trigger_rules", RULES_PATH)
trigger_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trigger_rules)

TriggerCondition = trigger_rules.TriggerCondition
TriggerRule = trigger_rules.TriggerRule


class TestTriggerCondition:
    """Tests for TriggerCondition class."""

    def test_init(self):
        """Test TriggerCondition initialization."""
        condition = TriggerCondition(
            field="player.status.hp",
            operator="lt",
            value=30
        )
        assert condition.field == "player.status.hp"
        assert condition.operator == "lt"
        assert condition.value == 30

    def test_to_dict(self):
        """Test to_dict method."""
        condition = TriggerCondition(
            field="player.status.hp",
            operator="lt",
            value=30
        )
        result = condition.to_dict()
        assert result == {
            "field": "player.status.hp",
            "operator": "lt",
            "value": 30
        }


class TestTriggerConditionEvaluate:
    """Tests for TriggerCondition.evaluate method."""

    @pytest.fixture
    def state(self) -> dict[str, Any]:
        """Create a test state."""
        return {
            "player": {
                "status": {
                    "hp": {"value": 25, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                    "shield": {"value": 50, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                    "is_knocked": {"value": False, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                }
            }
        }

    def test_evaluate_lt_true(self, state):
        """Test lt operator returns True when condition is met."""
        condition = TriggerCondition("player.status.hp", "lt", 30)
        assert condition.evaluate(state) is True

    def test_evaluate_lt_false(self, state):
        """Test lt operator returns False when condition is not met."""
        condition = TriggerCondition("player.status.hp", "lt", 20)
        assert condition.evaluate(state) is False

    def test_evaluate_gt_true(self, state):
        """Test gt operator returns True when condition is met."""
        condition = TriggerCondition("player.status.hp", "gt", 20)
        assert condition.evaluate(state) is True

    def test_evaluate_gt_false(self, state):
        """Test gt operator returns False when condition is not met."""
        condition = TriggerCondition("player.status.hp", "gt", 30)
        assert condition.evaluate(state) is False

    def test_evaluate_eq_true(self, state):
        """Test eq operator returns True when condition is met."""
        condition = TriggerCondition("player.status.hp", "eq", 25)
        assert condition.evaluate(state) is True

    def test_evaluate_eq_false(self, state):
        """Test eq operator returns False when condition is not met."""
        condition = TriggerCondition("player.status.hp", "eq", 30)
        assert condition.evaluate(state) is False

    def test_evaluate_lte_true(self, state):
        """Test lte operator returns True when condition is met."""
        condition = TriggerCondition("player.status.hp", "lte", 25)
        assert condition.evaluate(state) is True

    def test_evaluate_lte_boundary(self, state):
        """Test lte operator returns True at boundary."""
        condition = TriggerCondition("player.status.hp", "lte", 30)
        assert condition.evaluate(state) is True

    def test_evaluate_gte_true(self, state):
        """Test gte operator returns True when condition is met."""
        condition = TriggerCondition("player.status.hp", "gte", 25)
        assert condition.evaluate(state) is True

    def test_evaluate_gte_boundary(self, state):
        """Test gte operator returns True at boundary."""
        condition = TriggerCondition("player.status.hp", "gte", 20)
        assert condition.evaluate(state) is True

    def test_evaluate_ne_true(self, state):
        """Test ne operator returns True when condition is met."""
        condition = TriggerCondition("player.status.hp", "ne", 30)
        assert condition.evaluate(state) is True

    def test_evaluate_ne_false(self, state):
        """Test ne operator returns False when condition is not met."""
        condition = TriggerCondition("player.status.hp", "ne", 25)
        assert condition.evaluate(state) is False

    def test_evaluate_missing_field(self, state):
        """Test returns False for missing field."""
        condition = TriggerCondition("player.status.mana", "gt", 0)
        assert condition.evaluate(state) is False

    def test_evaluate_none_value(self, state):
        """Test returns False when field value is None."""
        state["player"]["status"]["hp"]["value"] = None
        condition = TriggerCondition("player.status.hp", "gt", 0)
        assert condition.evaluate(state) is False

    def test_evaluate_invalid_operator(self, state):
        """Test returns False for invalid operator - Pydantic validates at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TriggerCondition("player.status.hp", "invalid", 30)

    def test_evaluate_bool_field(self, state):
        """Test evaluating boolean field."""
        condition = TriggerCondition("player.status.is_knocked", "eq", False)
        assert condition.evaluate(state) is True


class TestTriggerRule:
    """Tests for TriggerRule class."""

    @pytest.fixture
    def conditions(self):
        """Create test conditions."""
        return [
            TriggerCondition("player.status.hp", "lt", 30)
        ]

    @pytest.fixture
    def templates(self):
        """Create test templates."""
        return {
            "combat": "Low HP! Heal!",
            "non_combat": "Your HP is low."
        }

    @pytest.fixture
    def rule(self, conditions, templates):
        """Create a test trigger rule."""
        return TriggerRule(
            rule_id="test_low_hp",
            name="Low HP Alert",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates=templates,
            cooldown_ms=5000
        )

    def test_init(self, rule):
        """Test TriggerRule initialization."""
        assert rule.id == "test_low_hp"
        assert rule.name == "Low HP Alert"
        assert rule.priority == 0
        assert rule.enabled is True
        assert rule.cooldown_ms == 5000
        assert rule.last_triggered_ms == 0

    def test_to_dict(self, rule):
        """Test to_dict method."""
        result = rule.to_dict()
        assert result["id"] == "test_low_hp"
        assert result["name"] == "Low HP Alert"
        assert result["priority"] == 0
        assert result["enabled"] is True
        assert len(result["conditions"]) == 1
        assert result["templates"]["combat"] == "Low HP! Heal!"

    def test_get_template_combat(self, rule):
        """Test get_template for combat state."""
        assert rule.get_template("combat") == "Low HP! Heal!"

    def test_get_template_non_combat(self, rule):
        """Test get_template for non_combat state."""
        assert rule.get_template("non_combat") == "Your HP is low."

    def test_get_template_missing(self, rule):
        """Test get_template returns None for missing state."""
        assert rule.get_template("invalid_state") is None

    def test_evaluate_enabled_true(self, rule):
        """Test evaluate returns True when conditions are met."""
        state = {
            "player": {
                "status": {
                    "hp": {"value": 25, "source": "test", "confidence": 1.0, "ts_ms": 12345}
                }
            }
        }
        assert rule.evaluate(state) is True

    def test_evaluate_enabled_false(self, rule):
        """Test evaluate returns False when conditions are not met."""
        state = {
            "player": {
                "status": {
                    "hp": {"value": 50, "source": "test", "confidence": 1.0, "ts_ms": 12345}
                }
            }
        }
        assert rule.evaluate(state) is False

    def test_evaluate_disabled(self, conditions, templates):
        """Test evaluate returns False when rule is disabled."""
        rule = TriggerRule(
            rule_id="test",
            name="Test",
            priority=0,
            enabled=False,
            conditions=conditions,
            templates=templates,
            cooldown_ms=5000
        )
        state = {
            "player": {
                "status": {
                    "hp": {"value": 25, "source": "test", "confidence": 1.0, "ts_ms": 12345}
                }
            }
        }
        assert rule.evaluate(state) is False


class TestTriggerRuleCooldown:
    """Tests for TriggerRule cooldown functionality."""

    @pytest.fixture
    def rule(self):
        """Create a test trigger rule."""
        return TriggerRule(
            rule_id="test",
            name="Test",
            priority=0,
            enabled=True,
            conditions=[],
            templates={"combat": "test"},
            cooldown_ms=5000
        )

    def test_is_on_cooldown_false_initial(self, rule):
        """Test is_on_cooldown returns False initially (before any trigger)."""
        # Use a time greater than cooldown_ms to avoid false positive
        # Since last_triggered_ms starts at 0, any time >= cooldown_ms should return False
        assert rule.is_on_cooldown(10000) is False

    def test_is_on_cooldown_true_after_trigger(self, rule):
        """Test is_on_cooldown returns True after triggering."""
        rule.update_last_triggered(10000)
        assert rule.is_on_cooldown(12000) is True  # 2000ms < 5000ms cooldown

    def test_is_on_cooldown_false_after_cooldown(self, rule):
        """Test is_on_cooldown returns False after cooldown period."""
        rule.update_last_triggered(10000)
        assert rule.is_on_cooldown(16000) is False  # 6000ms > 5000ms cooldown

    def test_update_last_triggered(self, rule):
        """Test update_last_triggered updates timestamp."""
        rule.update_last_triggered(12345)
        assert rule.last_triggered_ms == 12345


class TestTriggerRuleMultipleConditions:
    """Tests for TriggerRule with multiple conditions."""

    @pytest.fixture
    def multi_condition_rule(self):
        """Create a rule with multiple conditions."""
        conditions = [
            TriggerCondition("player.status.hp", "lt", 50),
            TriggerCondition("player.status.shield", "eq", 0),
        ]
        return TriggerRule(
            rule_id="vulnerable",
            name="Vulnerable Alert",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates={"combat": "No shield! Take cover!"},
            cooldown_ms=5000
        )

    def test_all_conditions_met(self, multi_condition_rule):
        """Test returns True when all conditions are met."""
        state = {
            "player": {
                "status": {
                    "hp": {"value": 30, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                    "shield": {"value": 0, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                }
            }
        }
        assert multi_condition_rule.evaluate(state) is True

    def test_one_condition_not_met(self, multi_condition_rule):
        """Test returns False when one condition is not met."""
        state = {
            "player": {
                "status": {
                    "hp": {"value": 30, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                    "shield": {"value": 50, "source": "test", "confidence": 1.0, "ts_ms": 12345},
                }
            }
        }
        assert multi_condition_rule.evaluate(state) is False
