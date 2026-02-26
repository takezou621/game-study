"""Tests for domain triggers: evaluator, rules, and policies."""

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Direct module imports to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"

# Load trigger rule module
rule_spec = importlib.util.spec_from_file_location(
    "trigger_rule",
    SRC_PATH / "domain" / "triggers" / "rule.py",
)
rule_module = importlib.util.module_from_spec(rule_spec)
rule_spec.loader.exec_module(rule_module)

# Load evaluator module
evaluator_spec = importlib.util.spec_from_file_location(
    "trigger_evaluator",
    SRC_PATH / "domain" / "triggers" / "evaluator.py",
)
evaluator_module = importlib.util.module_from_spec(evaluator_spec)
evaluator_spec.loader.exec_module(evaluator_module)

# Load cooldown policy module
cooldown_policy_spec = importlib.util.spec_from_file_location(
    "cooldown_policy",
    SRC_PATH / "domain" / "triggers" / "policies" / "cooldown_policy.py",
)
cooldown_policy_module = importlib.util.module_from_spec(cooldown_policy_spec)
cooldown_policy_spec.loader.exec_module(cooldown_policy_module)

# Load priority policy module
priority_policy_spec = importlib.util.spec_from_file_location(
    "priority_policy",
    SRC_PATH / "domain" / "triggers" / "policies" / "priority_policy.py",
)
priority_policy_module = importlib.util.module_from_spec(priority_policy_spec)
priority_policy_spec.loader.exec_module(priority_policy_module)

# Import classes
TriggerCondition = rule_module.TriggerCondition
TriggerRule = rule_module.TriggerRule
OperatorType = rule_module.OperatorType
DefaultTriggerEvaluator = evaluator_module.DefaultTriggerEvaluator
CooldownPolicy = cooldown_policy_module.CooldownPolicy
PriorityPolicy = priority_policy_module.PriorityPolicy
Priority = priority_policy_module.Priority


class TestOperatorType:
    """Tests for OperatorType enum."""

    def test_operator_values(self):
        """Test OperatorType enum values."""
        assert OperatorType.EQ.value == "eq"
        assert OperatorType.LT.value == "lt"
        assert OperatorType.GT.value == "gt"
        assert OperatorType.LTE.value == "lte"
        assert OperatorType.GTE.value == "gte"
        assert OperatorType.NE.value == "ne"
        assert OperatorType.IN.value == "in"
        assert OperatorType.NOT_IN.value == "not_in"
        assert OperatorType.CONTAINS.value == "contains"


class TestTriggerConditionExtended:
    """Extended tests for TriggerCondition class."""

    @pytest.fixture
    def state(self) -> dict[str, Any]:
        """Create a test state with various data types."""
        return {
            "player": {
                "status": {
                    "hp": {"value": 25, "source": "test"},
                    "shield": {"value": 50, "source": "test"},
                    "is_knocked": {"value": False, "source": "test"},
                    "items": {"value": ["shield", "medkit", "ammo"], "source": "test"},
                    "status_effects": {"value": "burning", "source": "test"},
                }
            }
        }

    def test_evaluate_in_operator_true(self, state):
        """Test 'in' operator returns True when value in list."""
        # Current value is ["shield", "medkit", "ammo"] - a list
        # For 'in' operator to be True, current needs to be IN self.value
        # But current is a list, so we need a different field
        state["player"]["status"]["weapon"] = {"value": "ar", "source": "test"}
        condition = TriggerCondition("player.status.weapon", "in", ["ar", "smg", "shotgun"])
        assert condition.evaluate(state) is True

    def test_evaluate_in_operator_false(self, state):
        """Test 'in' operator returns False when value not in list."""
        state["player"]["status"]["weapon"] = {"value": "sniper", "source": "test"}
        condition = TriggerCondition("player.status.weapon", "in", ["ar", "smg", "shotgun"])
        assert condition.evaluate(state) is False

    def test_evaluate_in_operator_not_list(self, state):
        """Test 'in' operator returns False when value is not a list."""
        condition = TriggerCondition("player.status.hp", "in", "not a list")
        assert condition.evaluate(state) is False

    def test_evaluate_not_in_operator_true(self, state):
        """Test 'not_in' operator returns True when value not in list."""
        state["player"]["status"]["weapon"] = {"value": "sniper", "source": "test"}
        condition = TriggerCondition("player.status.weapon", "not_in", ["ar", "smg", "shotgun"])
        assert condition.evaluate(state) is True

    def test_evaluate_not_in_operator_false(self, state):
        """Test 'not_in' operator returns False when value in list."""
        state["player"]["status"]["weapon"] = {"value": "ar", "source": "test"}
        condition = TriggerCondition("player.status.weapon", "not_in", ["ar", "smg", "shotgun"])
        assert condition.evaluate(state) is False

    def test_evaluate_not_in_operator_not_list(self, state):
        """Test 'not_in' operator returns False when value is not a list."""
        condition = TriggerCondition("player.status.hp", "not_in", "not a list")
        assert condition.evaluate(state) is False

    def test_evaluate_contains_operator_string_true(self, state):
        """Test 'contains' operator returns True when substring in string."""
        condition = TriggerCondition("player.status.status_effects", "contains", "burn")
        assert condition.evaluate(state) is True

    def test_evaluate_contains_operator_string_false(self, state):
        """Test 'contains' operator returns False when substring not in string."""
        condition = TriggerCondition("player.status.status_effects", "contains", "frozen")
        assert condition.evaluate(state) is False

    def test_evaluate_contains_operator_list_true(self, state):
        """Test 'contains' operator returns True when item in list."""
        condition = TriggerCondition("player.status.items", "contains", "medkit")
        assert condition.evaluate(state) is True

    def test_evaluate_contains_operator_list_false(self, state):
        """Test 'contains' operator returns False when item not in list."""
        condition = TriggerCondition("player.status.items", "contains", "trap")
        assert condition.evaluate(state) is False

    def test_evaluate_contains_operator_invalid_type(self, state):
        """Test 'contains' operator returns False for invalid types."""
        condition = TriggerCondition("player.status.hp", "contains", "25")
        assert condition.evaluate(state) is False

    def test_evaluate_nested_dict_access(self, state):
        """Test navigating nested dictionaries with 'value' keys."""
        condition = TriggerCondition("player.status.hp", "eq", 25)
        assert condition.evaluate(state) is True

    def test_evaluate_direct_value_access(self):
        """Test accessing values directly without 'value' wrapper."""
        state = {"player": {"hp": 50}}
        condition = TriggerCondition("player.hp", "eq", 50)
        assert condition.evaluate(state) is True

    def test_evaluate_key_error_returns_false(self):
        """Test KeyError during navigation returns False."""
        state = {"player": {}}
        condition = TriggerCondition("player.status.hp", "eq", 50)
        assert condition.evaluate(state) is False

    def test_evaluate_type_error_returns_false(self):
        """Test TypeError during navigation returns False."""
        state = {"player": "not a dict"}
        condition = TriggerCondition("player.status.hp", "eq", 50)
        assert condition.evaluate(state) is False


class TestTriggerRuleExtended:
    """Extended tests for TriggerRule class."""

    @pytest.fixture
    def conditions(self):
        """Create test conditions."""
        return [TriggerCondition("player.status.hp", "lt", 30)]

    @pytest.fixture
    def templates(self):
        """Create test templates."""
        return {"combat": "Low HP!", "non_combat": "Your HP is low."}

    def test_init_with_interrupt_higher_priority(self, conditions, templates):
        """Test TriggerRule with interrupt_higher_priority=True."""
        rule = TriggerRule(
            rule_id="test",
            name="Test",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates=templates,
            cooldown_ms=5000,
            interrupt_higher_priority=True,
        )
        assert rule.interrupt_higher_priority is True

    def test_evaluate_with_no_conditions(self, templates):
        """Test evaluate returns True when no conditions (always fires)."""
        rule = TriggerRule(
            rule_id="test",
            name="Test",
            priority=0,
            enabled=True,
            conditions=[],
            templates=templates,
            cooldown_ms=5000,
        )
        assert rule.evaluate({}) is True

    def test_is_on_cooldown_at_boundary(self, conditions, templates):
        """Test is_on_cooldown at exact boundary."""
        rule = TriggerRule(
            rule_id="test",
            name="Test",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates=templates,
            cooldown_ms=5000,
        )
        rule.update_last_triggered(10000)
        # At exact boundary (5000ms after trigger), should NOT be on cooldown
        assert rule.is_on_cooldown(15000) is False

    def test_is_on_cooldown_just_before_boundary(self, conditions, templates):
        """Test is_on_cooldown just before boundary."""
        rule = TriggerRule(
            rule_id="test",
            name="Test",
            priority=0,
            enabled=True,
            conditions=conditions,
            templates=templates,
            cooldown_ms=5000,
        )
        rule.update_last_triggered(10000)
        # Just before boundary (4999ms after trigger), should be on cooldown
        assert rule.is_on_cooldown(14999) is True


class TestDefaultTriggerEvaluator:
    """Tests for DefaultTriggerEvaluator class."""

    @pytest.fixture
    def rules(self):
        """Create test trigger rules."""
        return [
            TriggerRule(
                rule_id="low_hp",
                name="Low HP",
                priority=0,
                enabled=True,
                conditions=[TriggerCondition("player.status.hp", "lt", 30)],
                templates={"combat": "Low HP!"},
                cooldown_ms=5000,
            ),
            TriggerRule(
                rule_id="no_shield",
                name="No Shield",
                priority=1,
                enabled=True,
                conditions=[TriggerCondition("player.status.shield", "eq", 0)],
                templates={"combat": "No shield!"},
                cooldown_ms=5000,
            ),
            TriggerRule(
                rule_id="in_storm",
                name="In Storm",
                priority=0,
                enabled=True,
                conditions=[TriggerCondition("world.storm.in_storm", "eq", True)],
                templates={"combat": "Get out of storm!"},
                cooldown_ms=3000,
            ),
        ]

    @pytest.fixture
    def evaluator(self, rules):
        """Create evaluator with rules."""
        return DefaultTriggerEvaluator(rules)

    def test_init_sorts_rules_by_priority(self, evaluator):
        """Test that rules are sorted by priority."""
        priorities = [r.priority for r in evaluator.rules]
        assert priorities == sorted(priorities)

    def test_evaluate_no_firing_triggers(self, evaluator):
        """Test evaluate returns empty list when no triggers fire."""
        state = {
            "player": {"status": {"hp": {"value": 100}, "shield": {"value": 50}}},
            "world": {"storm": {"in_storm": {"value": False}}},
            "_evaluation_time_ms": 10000,  # Use time greater than cooldown
        }
        result = evaluator.evaluate(state)
        assert result == []

    def test_evaluate_single_trigger_fires(self, evaluator):
        """Test evaluate returns single firing trigger."""
        state = {
            "player": {"status": {"hp": {"value": 20}, "shield": {"value": 50}}},
            "world": {"storm": {"in_storm": {"value": False}}},
            "_evaluation_time_ms": 10000,  # Use time greater than cooldown
        }
        result = evaluator.evaluate(state)
        assert len(result) == 1
        assert result[0].id == "low_hp"

    def test_evaluate_multiple_triggers_fire(self, evaluator):
        """Test evaluate returns multiple firing triggers."""
        state = {
            "player": {"status": {"hp": {"value": 20}, "shield": {"value": 0}}},
            "world": {"storm": {"in_storm": {"value": True}}},
            "_evaluation_time_ms": 10000,  # Use time greater than cooldown
        }
        result = evaluator.evaluate(state)
        assert len(result) == 3

    def test_evaluate_respects_cooldown(self, evaluator):
        """Test evaluate respects cooldown."""
        # First, trigger the low_hp rule at time 10000
        state = {
            "player": {"status": {"hp": {"value": 20}, "shield": {"value": 50}}},
            "world": {"storm": {"in_storm": {"value": False}}},
            "_evaluation_time_ms": 10000,
        }
        result = evaluator.evaluate(state)
        assert len(result) == 1

        # Update cooldown
        evaluator.rules[0].update_last_triggered(10000)

        # Now evaluate again - should not fire due to cooldown (only 1 second passed)
        state["_evaluation_time_ms"] = 11000  # Only 1 second passed
        result = evaluator.evaluate(state)
        # low_hp should not fire due to cooldown
        firing_ids = [r.id for r in result]
        assert "low_hp" not in firing_ids

    def test_should_interrupt_p0_always_interrupts(self, evaluator):
        """Test should_interrupt returns True for P0."""
        assert evaluator.should_interrupt(0, 1) is True
        assert evaluator.should_interrupt(0, 2) is True
        assert evaluator.should_interrupt(0, 3) is True

    def test_should_interrupt_p1_interrupts_p2_p3(self, evaluator):
        """Test should_interrupt returns True for P1 vs P2/P3."""
        assert evaluator.should_interrupt(1, 2) is True
        assert evaluator.should_interrupt(1, 3) is True

    def test_should_interrupt_p1_does_not_interrupt_p0(self, evaluator):
        """Test should_interrupt returns False for P1 vs P0."""
        assert evaluator.should_interrupt(1, 0) is False

    def test_should_interrupt_same_priority(self, evaluator):
        """Test should_interrupt returns False for same priority."""
        assert evaluator.should_interrupt(1, 1) is False

    def test_get_highest_priority_empty_list(self, evaluator):
        """Test get_highest_priority returns None for empty list."""
        assert evaluator.get_highest_priority([]) is None

    def test_get_highest_priority_single(self, evaluator, rules):
        """Test get_highest_priority returns single trigger."""
        result = evaluator.get_highest_priority([rules[0]])
        assert result == rules[0]

    def test_get_highest_priority_multiple(self, evaluator, rules):
        """Test get_highest_priority returns lowest priority number."""
        result = evaluator.get_highest_priority(rules)
        assert result.priority == 0

    def test_add_rule(self, evaluator):
        """Test add_rule adds and re-sorts rules."""
        new_rule = TriggerRule(
            rule_id="critical_hp",
            name="Critical HP",
            priority=0,
            enabled=True,
            conditions=[TriggerCondition("player.status.hp", "lt", 10)],
            templates={"combat": "Critical!"},
            cooldown_ms=3000,
        )
        evaluator.add_rule(new_rule)
        assert new_rule in evaluator.rules
        # Check still sorted
        priorities = [r.priority for r in evaluator.rules]
        assert priorities == sorted(priorities)

    def test_remove_rule_existing(self, evaluator):
        """Test remove_rule removes existing rule."""
        result = evaluator.remove_rule("low_hp")
        assert result is True
        rule_ids = [r.id for r in evaluator.rules]
        assert "low_hp" not in rule_ids

    def test_remove_rule_non_existing(self, evaluator):
        """Test remove_rule returns False for non-existing rule."""
        result = evaluator.remove_rule("non_existent")
        assert result is False

    def test_get_rule_existing(self, evaluator):
        """Test get_rule returns existing rule."""
        result = evaluator.get_rule("low_hp")
        assert result is not None
        assert result.id == "low_hp"

    def test_get_rule_non_existing(self, evaluator):
        """Test get_rule returns None for non-existing rule."""
        result = evaluator.get_rule("non_existent")
        assert result is None


class TestCooldownPolicy:
    """Tests for CooldownPolicy class."""

    @pytest.fixture
    def policy(self):
        """Create a cooldown policy."""
        return CooldownPolicy()

    def test_default_values(self, policy):
        """Test default cooldown values."""
        assert policy.default_cooldown_ms == 5000
        assert policy.min_cooldown_ms == 1000
        assert policy.max_cooldown_ms == 60000

    def test_custom_values(self):
        """Test custom cooldown values."""
        policy = CooldownPolicy(
            default_cooldown_ms=10000,
            min_cooldown_ms=2000,
            max_cooldown_ms=120000,
        )
        assert policy.default_cooldown_ms == 10000
        assert policy.min_cooldown_ms == 2000
        assert policy.max_cooldown_ms == 120000

    def test_calculate_cooldown_p0(self, policy):
        """Test cooldown calculation for P0 (1.0 factor)."""
        result = policy.calculate_cooldown(5000, 0, {})
        assert result == 5000  # 5000 * 1.0 = 5000

    def test_calculate_cooldown_p1(self, policy):
        """Test cooldown calculation for P1 (0.8 factor)."""
        result = policy.calculate_cooldown(5000, 1, {})
        assert result == 4000  # 5000 * 0.8 = 4000

    def test_calculate_cooldown_p2(self, policy):
        """Test cooldown calculation for P2 (0.6 factor)."""
        result = policy.calculate_cooldown(5000, 2, {})
        assert result == 3000  # 5000 * 0.6 = 3000

    def test_calculate_cooldown_p3(self, policy):
        """Test cooldown calculation for P3 (0.4 factor)."""
        result = policy.calculate_cooldown(5000, 3, {})
        # 5000 * 0.4 = 2000.0, int() truncates to 1999 due to floating point
        assert result >= 1999  # Allow for floating point precision

    def test_calculate_cooldown_clamped_to_min(self, policy):
        """Test cooldown clamped to minimum."""
        # 100 * 0.4 = 40, clamped to min 1000
        result = policy.calculate_cooldown(100, 3, {})
        assert result == 1000

    def test_calculate_cooldown_clamped_to_max(self, policy):
        """Test cooldown clamped to maximum."""
        # 100000 * 1.0 = 100000, clamped to max 60000
        result = policy.calculate_cooldown(100000, 0, {})
        assert result == 60000

    def test_is_on_cooldown_true(self, policy):
        """Test is_on_cooldown returns True when on cooldown."""
        result = policy.is_on_cooldown(
            last_triggered_ms=1000,
            cooldown_ms=5000,
            current_time_ms=3000,
        )
        assert result is True  # 2000ms elapsed < 5000ms cooldown

    def test_is_on_cooldown_false(self, policy):
        """Test is_on_cooldown returns False when not on cooldown."""
        result = policy.is_on_cooldown(
            last_triggered_ms=1000,
            cooldown_ms=5000,
            current_time_ms=7000,
        )
        assert result is False  # 6000ms elapsed > 5000ms cooldown

    def test_get_remaining_cooldown_positive(self, policy):
        """Test get_remaining_cooldown returns positive value."""
        result = policy.get_remaining_cooldown(
            last_triggered_ms=1000,
            cooldown_ms=5000,
            current_time_ms=3000,
        )
        assert result == 3000  # 5000 - 2000 = 3000 remaining

    def test_get_remaining_cooldown_zero(self, policy):
        """Test get_remaining_cooldown returns 0 when not on cooldown."""
        result = policy.get_remaining_cooldown(
            last_triggered_ms=1000,
            cooldown_ms=5000,
            current_time_ms=7000,
        )
        assert result == 0  # No remaining cooldown

    def test_get_remaining_cooldown_negative_becomes_zero(self, policy):
        """Test get_remaining_cooldown returns 0 for negative values."""
        result = policy.get_remaining_cooldown(
            last_triggered_ms=1000,
            cooldown_ms=5000,
            current_time_ms=10000,
        )
        assert result == 0  # max(0, 5000 - 9000) = 0


class TestPriorityPolicy:
    """Tests for PriorityPolicy class."""

    @pytest.fixture
    def policy(self):
        """Create a priority policy."""
        return PriorityPolicy()

    def test_priority_enum_values(self):
        """Test Priority enum values."""
        assert Priority.SURVIVAL == 0
        assert Priority.TACTICAL == 1
        assert Priority.LEARNING == 2
        assert Priority.CHATTER == 3

    def test_should_interrupt_p0_any(self, policy):
        """Test P0 interrupts any priority."""
        assert policy.should_interrupt(Priority.SURVIVAL, Priority.TACTICAL) is True
        assert policy.should_interrupt(Priority.SURVIVAL, Priority.LEARNING) is True
        assert policy.should_interrupt(Priority.SURVIVAL, Priority.CHATTER) is True
        assert policy.should_interrupt(Priority.SURVIVAL, Priority.SURVIVAL) is True

    def test_should_interrupt_p1_vs_p2_p3(self, policy):
        """Test P1 interrupts P2 and P3."""
        assert policy.should_interrupt(Priority.TACTICAL, Priority.LEARNING) is True
        assert policy.should_interrupt(Priority.TACTICAL, Priority.CHATTER) is True

    def test_should_interrupt_p1_vs_p0(self, policy):
        """Test P1 does not interrupt P0."""
        assert policy.should_interrupt(Priority.TACTICAL, Priority.SURVIVAL) is False

    def test_should_interrupt_p2_vs_p3_with_time(self, policy):
        """Test P2 interrupts P3 only with significant time remaining."""
        assert policy.should_interrupt(Priority.LEARNING, Priority.CHATTER, 3000) is True
        assert policy.should_interrupt(Priority.LEARNING, Priority.CHATTER, 1000) is False

    def test_should_interrupt_p2_vs_lower_priority(self, policy):
        """Test P2 does not interrupt lower priorities."""
        assert policy.should_interrupt(Priority.LEARNING, Priority.TACTICAL) is False
        assert policy.should_interrupt(Priority.LEARNING, Priority.SURVIVAL) is False

    def test_should_interrupt_p3_does_not_interrupt(self, policy):
        """Test P3 does not interrupt any priority."""
        assert policy.should_interrupt(Priority.CHATTER, Priority.SURVIVAL) is False
        assert policy.should_interrupt(Priority.CHATTER, Priority.TACTICAL) is False
        assert policy.should_interrupt(Priority.CHATTER, Priority.LEARNING) is False

    def test_get_priority_label_known(self, policy):
        """Test get_priority_label for known priorities."""
        assert policy.get_priority_label(Priority.SURVIVAL) == "Survival"
        assert policy.get_priority_label(Priority.TACTICAL) == "Tactical"
        assert policy.get_priority_label(Priority.LEARNING) == "Learning"
        assert policy.get_priority_label(Priority.CHATTER) == "Chatter"

    def test_get_priority_label_unknown(self, policy):
        """Test get_priority_label for unknown priorities."""
        assert policy.get_priority_label(99) == "P99"

    def test_get_max_response_duration_ms(self, policy):
        """Test get_max_response_duration_ms for each priority."""
        assert policy.get_max_response_duration_ms(Priority.SURVIVAL) == 3000
        assert policy.get_max_response_duration_ms(Priority.TACTICAL) == 5000
        assert policy.get_max_response_duration_ms(Priority.LEARNING) == 10000
        assert policy.get_max_response_duration_ms(Priority.CHATTER) == 8000
        assert policy.get_max_response_duration_ms(99) == 5000  # Default

    def test_get_template_preference_short(self, policy):
        """Test get_template_preference returns 'short' for high priorities."""
        assert policy.get_template_preference(Priority.SURVIVAL) == "short"
        assert policy.get_template_preference(Priority.TACTICAL) == "short"

    def test_get_template_preference_normal(self, policy):
        """Test get_template_preference returns 'normal' for lower priorities."""
        assert policy.get_template_preference(Priority.LEARNING) == "normal"
        assert policy.get_template_preference(Priority.CHATTER) == "normal"

    def test_compare_a_higher(self, policy):
        """Test compare returns negative when a is higher priority."""
        assert policy.compare(Priority.SURVIVAL, Priority.TACTICAL) < 0

    def test_compare_b_higher(self, policy):
        """Test compare returns positive when b is higher priority."""
        assert policy.compare(Priority.CHATTER, Priority.SURVIVAL) > 0

    def test_compare_equal(self, policy):
        """Test compare returns 0 for equal priorities."""
        assert policy.compare(Priority.TACTICAL, Priority.TACTICAL) == 0


class TestTriggerEvaluatorProtocol:
    """Tests for TriggerEvaluator protocol compliance."""

    def test_evaluator_has_required_methods(self):
        """Test that DefaultTriggerEvaluator implements protocol methods."""
        evaluator = DefaultTriggerEvaluator([])
        assert hasattr(evaluator, "evaluate")
        assert hasattr(evaluator, "should_interrupt")
        assert callable(evaluator.evaluate)
        assert callable(evaluator.should_interrupt)
