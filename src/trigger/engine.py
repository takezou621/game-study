"""Trigger engine for evaluating and triggering responses."""

import yaml
from typing import Dict, Any, List, Optional
from utils.time import get_timestamp_ms
from trigger.rules import TriggerRule, TriggerCondition


class TriggerEngine:
    """Engine for evaluating triggers and managing priority/cooldown."""

    def __init__(self, triggers_config_path: str):
        """
        Initialize trigger engine.

        Args:
            triggers_config_path: Path to triggers YAML configuration
        """
        self.config = self._load_config(triggers_config_path)
        self.rules = self._parse_rules()

        # Settings
        self.settings = self.config.get('settings', {})
        self.cooldown_enabled = self.settings.get('cooldown_enabled', True)
        self.interrupt_higher_priority = self.settings.get('interrupt_higher_priority', True)
        self.max_response_length = self.settings.get('max_response_length_chars', 200)
        self.combat_suppress_priority = self.settings.get('combat_suppress_priority', [2, 3])
        self.inactivity_threshold_ms = self.settings.get('inactivity_threshold_ms', 30000)

        # State
        self.last_trigger_time_ms = 0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load triggers configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _parse_rules(self) -> List[TriggerRule]:
        """Parse triggers configuration into TriggerRule objects."""
        rules = []
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
                templates=trigger.get('template', {}),
                cooldown_ms=trigger.get('cooldown_ms', 0),
                interrupt_higher_priority=trigger.get('interrupt_higher_priority', False),
            )
            rules.append(rule)

        # Sort by priority (0 = highest)
        rules.sort(key=lambda r: r.priority)
        return rules

    def evaluate_triggers(
        self,
        state: Dict[str, Any],
        movement_state: str
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate triggers and return the highest priority triggered rule.

        Args:
            state: Current game state
            movement_state: Movement state ("combat" or "non_combat")

        Returns:
            Trigger result with rule, template, or None if no trigger
        """
        current_time_ms = get_timestamp_ms()
        inactivity_duration_ms = current_time_ms - self.last_trigger_time_ms

        # Update inactivity in state
        state["session"]["inactivity_duration_ms"]["value"] = inactivity_duration_ms

        triggered_rules = []

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

    def get_rule_by_id(self, rule_id: str) -> Optional[TriggerRule]:
        """
        Get rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            TriggerRule or None if not found
        """
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a rule.

        Args:
            rule_id: Rule identifier

        Returns:
            True if rule was enabled
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
            rule_id: Rule identifier

        Returns:
            True if rule was disabled
        """
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """
        Get all rules as dictionaries.

        Returns:
            List of rule dictionaries
        """
        return [rule.to_dict() for rule in self.rules]
