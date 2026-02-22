"""Trigger engine for evaluating and triggering responses."""

from typing import Any

import yaml

from trigger.rules import TriggerCondition, TriggerRule
from utils.time import get_timestamp_ms


class TriggerEngine:
    """
    Engine for evaluating triggers and managing priority/cooldown.

    This engine loads trigger rules from a YAML configuration and evaluates
    them against the current game state. It handles:
        - Priority-based trigger selection (0 = highest priority)
        - Cooldown management between triggers
        - Combat state suppression of lower-priority learning triggers
        - Movement state template selection (combat vs non_combat)

    Attributes:
        config: Raw configuration dictionary from YAML.
        rules: List of TriggerRule objects sorted by priority.
        settings: Configuration settings dict.
        cooldown_enabled: Whether cooldown timers are enforced.
        interrupt_higher_priority: Whether higher priority triggers can interrupt.
        max_response_length: Maximum character limit for responses.
        combat_suppress_priority: Priority levels to suppress during combat.
        inactivity_threshold_ms: Inactivity threshold in milliseconds.
        last_trigger_time_ms: Timestamp of the last triggered rule.
    """

    def __init__(self, triggers_config_path: str) -> None:
        """
        Initialize trigger engine.

        Args:
            triggers_config_path: Path to triggers YAML configuration file.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        self.config: dict[str, Any] = self._load_config(triggers_config_path)
        self.rules: list[TriggerRule] = self._parse_rules()

        # Settings
        self.settings: dict[str, Any] = self.config.get('settings', {})
        self.cooldown_enabled: bool = self.settings.get('cooldown_enabled', True)
        self.interrupt_higher_priority: bool = self.settings.get('interrupt_higher_priority', True)
        self.max_response_length: int = self.settings.get('max_response_length_chars', 200)
        self.combat_suppress_priority: list[int] = self.settings.get('combat_suppress_priority', [2, 3])
        self.inactivity_threshold_ms: int = self.settings.get('inactivity_threshold_ms', 30000)

        # State
        self.last_trigger_time_ms: int = 0

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """
        Load triggers configuration from YAML.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Dictionary containing the parsed YAML configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _parse_rules(self) -> list[TriggerRule]:
        """
        Parse triggers configuration into TriggerRule objects.

        Reads trigger definitions from the configuration and creates
        TriggerRule instances with their associated conditions.

        Returns:
            List of TriggerRule objects sorted by priority (0 = highest).
        """
        rules: list[TriggerRule] = []
        triggers = self.config.get('triggers', [])

        for trigger in triggers:
            rule = TriggerRule(
                rule_id=trigger['id'],
                name=trigger['name'],
                priority=trigger['priority'],
                enabled=trigger.get('enabled', True),
                conditions=[
                    TriggerCondition(c['field'], c['operator'], c['value'])
                    for c in trigger.get('conditions', [])
                ],
                templates=trigger.get('templates') or trigger.get('template', {}),
                cooldown_ms=trigger.get('cooldown_ms', 0),
                interrupt_higher_priority=trigger.get('interrupt_higher_priority', False),
            )
            rules.append(rule)

        # Sort by priority (0 = highest)
        rules.sort(key=lambda r: r.priority)
        return rules

    def evaluate_triggers(
        self,
        state: dict[str, Any],
        movement_state: str
    ) -> dict[str, Any] | None:
        """
        Evaluate triggers and return the highest priority triggered rule.

        Evaluates all enabled rules against the current state, applying
        cooldown, combat suppression, and template filtering.

        Args:
            state: Current game state dictionary. Expected to contain
                nested keys like "player.status.hp", "world.storm.phase", etc.
            movement_state: Movement state, either "combat" or "non_combat".

        Returns:
            Trigger result dictionary containing:
                - rule_id (str): ID of the triggered rule
                - rule_name (str): Name of the triggered rule
                - priority (int): Priority level of the rule
                - template (str): Response template for the movement state
                - movement_state (str): The movement state used
                - timestamp_ms (int): When the trigger was evaluated
            Returns None if no rules were triggered.
        """
        current_time_ms = get_timestamp_ms()
        inactivity_duration_ms = current_time_ms - self.last_trigger_time_ms

        # Update inactivity in state
        state["session"]["inactivity_duration_ms"]["value"] = inactivity_duration_ms

        triggered_rules: list[tuple[TriggerRule, str]] = []

        # Evaluate all rules
        for rule in self.rules:
            # Check if rule is enabled and conditions are met
            if not rule.enabled:
                continue

            if not rule.evaluate(state):
                continue

            # Check cooldown
            if self.cooldown_enabled and rule.is_on_cooldown(current_time_ms):
                continue

            # Check combat suppression
            if movement_state == "combat" and rule.priority in self.combat_suppress_priority:
                continue

            # Get template for movement state
            template = rule.get_template(movement_state)
            if template is None:
                continue

            triggered_rules.append((rule, template))

        if not triggered_rules:
            return None

        # Return highest priority rule (already sorted)
        rule, template = triggered_rules[0]

        # Update last triggered time
        rule.update_last_triggered(current_time_ms)
        self.last_trigger_time_ms = current_time_ms

        return {
            "rule_id": rule.id,
            "rule_name": rule.name,
            "priority": rule.priority,
            "template": template,
            "movement_state": movement_state,
            "timestamp_ms": current_time_ms,
        }

    def get_rule_by_id(self, rule_id: str) -> TriggerRule | None:
        """
        Get rule by ID.

        Args:
            rule_id: Rule identifier string.

        Returns:
            TriggerRule object if found, None otherwise.
        """
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a rule.

        Args:
            rule_id: Rule identifier string.

        Returns:
            True if the rule was found and enabled, False otherwise.
        """
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable a rule.

        Args:
            rule_id: Rule identifier string.

        Returns:
            True if the rule was found and disabled, False otherwise.
        """
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    def get_all_rules(self) -> list[dict[str, Any]]:
        """
        Get all rules as dictionaries.

        Returns:
            List of rule dictionaries containing id, name, priority,
            enabled status, and other rule properties.
        """
        return [rule.to_dict() for rule in self.rules]
