"""Tests for application use cases: EvaluateTriggersUseCase."""

import importlib.util
from pathlib import Path

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

# Load priority policy module
priority_policy_spec = importlib.util.spec_from_file_location(
    "priority_policy",
    SRC_PATH / "domain" / "triggers" / "policies" / "priority_policy.py",
)
priority_policy_module = importlib.util.module_from_spec(priority_policy_spec)
priority_policy_spec.loader.exec_module(priority_policy_module)

# Load evaluate_triggers use case
evaluate_triggers_spec = importlib.util.spec_from_file_location(
    "evaluate_triggers",
    SRC_PATH / "application" / "usecases" / "evaluate_triggers.py",
)
evaluate_triggers_module = importlib.util.module_from_spec(evaluate_triggers_spec)
evaluate_triggers_spec.loader.exec_module(evaluate_triggers_module)

# Load trigger DTO module
trigger_dto_spec = importlib.util.spec_from_file_location(
    "trigger_dto",
    SRC_PATH / "application" / "dto" / "trigger_dto.py",
)
trigger_dto_module = importlib.util.module_from_spec(trigger_dto_spec)
trigger_dto_spec.loader.exec_module(trigger_dto_module)

# Import classes
TriggerCondition = rule_module.TriggerCondition
TriggerRule = rule_module.TriggerRule
DefaultTriggerEvaluator = evaluator_module.DefaultTriggerEvaluator
PriorityPolicy = priority_policy_module.PriorityPolicy
Priority = priority_policy_module.Priority
EvaluateTriggersUseCase = evaluate_triggers_module.EvaluateTriggersUseCase
TriggerEvaluationDTO = trigger_dto_module.TriggerEvaluationDTO


class TestEvaluateTriggersUseCase:
    """Tests for EvaluateTriggersUseCase."""

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
                templates={"combat": "Low HP!", "non_combat": "Heal up!"},
                cooldown_ms=5000,
            ),
            TriggerRule(
                rule_id="no_shield",
                name="No Shield",
                priority=1,
                enabled=True,
                conditions=[TriggerCondition("player.status.shield", "eq", 0)],
                templates={"combat": "No shield!", "non_combat": "Get shields!"},
                cooldown_ms=5000,
            ),
        ]

    @pytest.fixture
    def evaluator(self, rules):
        """Create evaluator with rules."""
        return DefaultTriggerEvaluator(rules)

    @pytest.fixture
    def priority_policy(self):
        """Create priority policy."""
        return PriorityPolicy()

    @pytest.fixture
    def usecase(self, evaluator, priority_policy):
        """Create use case instance."""
        return EvaluateTriggersUseCase(
            evaluator=evaluator,
            priority_policy=priority_policy,
        )

    @pytest.fixture
    def state_low_hp(self):
        """Create state with low HP."""
        return {
            "player": {"status": {"hp": {"value": 20}, "shield": {"value": 50}}},
        }

    @pytest.fixture
    def state_healthy(self):
        """Create healthy state."""
        return {
            "player": {"status": {"hp": {"value": 100}, "shield": {"value": 50}}},
        }

    def test_init(self, usecase):
        """Test use case initialization."""
        assert usecase.evaluator is not None
        assert usecase.priority_policy is not None
        assert usecase.current_speech_priority is None

    def test_execute_no_firing_triggers(self, usecase, state_healthy):
        """Test execute with no firing triggers."""
        input_dto = TriggerEvaluationDTO(
            state=state_healthy,
            movement_state="non_combat",
            evaluation_time_ms=1000,
        )
        result = usecase.execute(input_dto)
        assert result.has_firing_triggers is False
        assert result.selected_trigger is None
        assert result.suppressed_count == 0

    def test_execute_single_firing_trigger(self, usecase, state_low_hp):
        """Test execute with single firing trigger."""
        input_dto = TriggerEvaluationDTO(
            state=state_low_hp,
            movement_state="combat",
            evaluation_time_ms=10000,  # Use time > cooldown to avoid initial cooldown
        )
        result = usecase.execute(input_dto)
        assert result.has_firing_triggers is True
        assert result.selected_trigger is not None
        assert result.selected_trigger.trigger_id == "low_hp"
        assert result.selected_trigger.template == "Low HP!"
        assert result.suppressed_count == 0

    def test_execute_multiple_firing_triggers(self, usecase):
        """Test execute with multiple firing triggers."""
        state = {
            "player": {"status": {"hp": {"value": 20}, "shield": {"value": 0}}},
        }
        input_dto = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=10000,  # Use time > cooldown to avoid initial cooldown
        )
        result = usecase.execute(input_dto)
        assert result.has_firing_triggers is True
        assert len(result.firing_triggers) == 2
        # Highest priority (lowest number) should be selected
        assert result.selected_trigger.priority == 0
        assert result.suppressed_count == 1

    def test_execute_with_interrupt_check(self, usecase, state_low_hp):
        """Test execute checks for interrupt when active speech priority set."""
        input_dto = TriggerEvaluationDTO(
            state=state_low_hp,
            movement_state="combat",
            evaluation_time_ms=10000,  # Use time > cooldown to avoid initial cooldown
            active_speech_priority=Priority.CHATTER,  # P3
        )
        result = usecase.execute(input_dto)
        # P0 (low_hp) should interrupt P3 (chatter)
        assert result.selected_trigger.should_interrupt is True

    def test_execute_no_interrupt_same_priority(self, usecase):
        """Test execute does not set interrupt for same priority."""
        state = {
            "player": {"status": {"hp": {"value": 100}, "shield": {"value": 0}}},
        }
        input_dto = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=10000,  # Use time > cooldown to avoid initial cooldown
            active_speech_priority=Priority.TACTICAL,  # P1 same as no_shield
        )
        result = usecase.execute(input_dto)
        # P1 should not interrupt P1
        assert result.selected_trigger.should_interrupt is False

    def test_execute_sorts_triggers_by_priority(self, usecase):
        """Test execute sorts triggers by priority."""
        state = {
            "player": {"status": {"hp": {"value": 20}, "shield": {"value": 0}}},
        }
        input_dto = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=10000,  # Use time > cooldown to avoid initial cooldown
        )
        result = usecase.execute(input_dto)
        priorities = [t.priority for t in result.firing_triggers]
        assert priorities == sorted(priorities)

    def test_set_current_speech_priority(self, usecase):
        """Test set_current_speech_priority."""
        usecase.set_current_speech_priority(2)
        assert usecase.current_speech_priority == 2

        usecase.set_current_speech_priority(None)
        assert usecase.current_speech_priority is None

    def test_mark_triggered(self, usecase, rules):
        """Test mark_triggered updates cooldown."""
        rule = rules[0]
        usecase.mark_triggered(rule)
        assert rule.last_triggered_ms > 0

    def test_create_factory_method(self, rules):
        """Test create factory method."""
        usecase = EvaluateTriggersUseCase.create(rules)
        assert usecase.evaluator is not None
        assert usecase.priority_policy is not None
        assert len(usecase.evaluator.rules) == 2


class TestEvaluateTriggersUseCaseWithCooldown:
    """Tests for EvaluateTriggersUseCase with cooldown scenarios."""

    @pytest.fixture
    def rules(self):
        """Create test trigger rules."""
        return [
            TriggerRule(
                rule_id="test_rule",
                name="Test Rule",
                priority=0,
                enabled=True,
                conditions=[TriggerCondition("player.status.hp", "lt", 50)],
                templates={"combat": "Low HP!"},
                cooldown_ms=10000,
            ),
        ]

    @pytest.fixture
    def usecase(self, rules):
        """Create use case instance."""
        return EvaluateTriggersUseCase.create(rules)

    def test_execute_respects_cooldown(self, usecase):
        """Test execute respects cooldown after marking triggered."""
        state = {"player": {"status": {"hp": {"value": 30}}}}

        # First evaluation - should fire (use time > initial cooldown)
        input_dto = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=10000,  # Use time > cooldown
        )
        result = usecase.execute(input_dto)
        assert result.has_firing_triggers is True

        # Mark as triggered
        rule = usecase.evaluator.get_rule("test_rule")
        usecase.mark_triggered(rule)

        # Second evaluation right after - should not fire due to cooldown
        input_dto2 = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=11000,  # Only 1 second after trigger
        )
        result2 = usecase.execute(input_dto2)
        assert result2.has_firing_triggers is False

    def test_execute_fires_after_cooldown(self, usecase):
        """Test execute fires after cooldown period."""
        state = {"player": {"status": {"hp": {"value": 30}}}}

        # First evaluation (use time > initial cooldown)
        input_dto = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=10000,
        )
        result = usecase.execute(input_dto)
        assert result.has_firing_triggers is True

        # Mark as triggered directly updating the rule's last_triggered_ms
        rule = usecase.evaluator.get_rule("test_rule")
        rule.update_last_triggered(10000)

        # Second evaluation after cooldown period
        input_dto2 = TriggerEvaluationDTO(
            state=state,
            movement_state="combat",
            evaluation_time_ms=25000,  # 15 seconds after trigger (> 10000 cooldown)
        )
        result2 = usecase.execute(input_dto2)
        assert result2.has_firing_triggers is True


class TestTriggerEvaluationDTO:
    """Tests for TriggerEvaluationDTO."""

    def test_init_defaults(self):
        """Test TriggerEvaluationDTO initialization with defaults."""
        dto = TriggerEvaluationDTO(
            state={"player": {"status": {"hp": {"value": 100}}}},
            movement_state="non_combat",
            evaluation_time_ms=1000,
        )
        assert dto.state is not None
        assert dto.movement_state == "non_combat"
        assert dto.evaluation_time_ms == 1000
        assert dto.active_speech_priority is None

    def test_init_with_values(self):
        """Test TriggerEvaluationDTO initialization with values."""
        dto = TriggerEvaluationDTO(
            state={"player": {"status": {"hp": {"value": 50}}}},
            movement_state="combat",
            evaluation_time_ms=2000,
            active_speech_priority=1,
        )
        assert dto.active_speech_priority == 1


class TestTriggerResultDTO:
    """Tests for TriggerResultDTO."""

    def test_init_defaults(self):
        """Test TriggerResultDTO initialization."""
        from application.dto.trigger_dto import TriggerResultDTO

        dto = TriggerResultDTO(
            trigger_id="test",
            trigger_name="Test Trigger",
            priority=0,
            template="Test template",
            movement_state="combat",
            should_interrupt=False,
            confidence=1.0,
            evaluation_time_ms=1000,
        )
        assert dto.trigger_id == "test"
        assert dto.trigger_name == "Test Trigger"
        assert dto.priority == 0
        assert dto.template == "Test template"
        assert dto.movement_state == "combat"
        assert dto.should_interrupt is False
        assert dto.confidence == 1.0


class TestTriggerEvaluationResultDTO:
    """Tests for TriggerEvaluationResultDTO."""

    def test_empty_result(self):
        """Test empty result."""
        from application.dto.trigger_dto import TriggerEvaluationResultDTO

        result = TriggerEvaluationResultDTO(
            firing_triggers=[],
            selected_trigger=None,
            suppressed_count=0,
        )
        assert result.has_firing_triggers is False
        assert result.selected_trigger is None
        assert result.firing_triggers == []

    def test_result_with_triggers(self):
        """Test result with triggers."""
        from application.dto.trigger_dto import TriggerEvaluationResultDTO, TriggerResultDTO

        trigger = TriggerResultDTO(
            trigger_id="test",
            trigger_name="Test",
            priority=0,
            template="Test",
            movement_state="combat",
            should_interrupt=False,
            confidence=1.0,
            evaluation_time_ms=1000,
        )
        result = TriggerEvaluationResultDTO(
            firing_triggers=[trigger],
            selected_trigger=trigger,
            suppressed_count=0,
        )
        assert result.has_firing_triggers is True
        assert result.selected_trigger == trigger
