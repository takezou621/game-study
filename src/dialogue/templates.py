"""Dialogue template management for AI coach responses."""

import os
from typing import Dict, Optional, Any


class DialogueTemplateManager:
    """Manage dialogue templates for different situations."""

    def __init__(self):
        """Initialize template manager."""
        self.templates = {
            # P0: Survival (Combat)
            "p0_knocked_combat": "You're knocked! Ping your location!",
            "p0_knocked_non_combat": "Stay calm! Your teammates are coming to revive you.",
            "p0_low_hp_combat": "Low HP! Find cover immediately!",
            "p0_low_hp_non_combat": "Your health is critical. Heal up before fighting.",
            "p0_storm_damage_combat": "Get out of the storm! Now!",
            "p0_storm_damage_non_combat": "You're taking storm damage. Move to the safe zone.",

            # P1: Tactical
            "p1_storm_shrinking_combat": "Storm is moving! Get ready to rotate.",
            "p1_storm_shrinking_non_combat": "The storm is shrinking. Time to move to the safe zone.",
            "p1_rotate_combat": "Next circle is close. Rotate safely.",
            "p1_rotate_non_combat": "You're far from the next circle. Start moving that way.",

            # P2: Learning
            "p2_weapon_non_combat": "You picked up a {weapon_name}! This is great for {situation}.",
            "p2_item_non_combat": "Nice find! This {item_name} is useful for {purpose}.",
            "p2_past_tense_non_combat": "What happened in that fight? Tell me what you did.",

            # P3: Chatter
            "p3_session_summary": "Great session! What did you learn today?",
            "p3_small_talk": "How's it going? Anything you want to practice?",
        }

    def get_template(
        self,
        trigger_id: str,
        movement_state: str,
        **kwargs
    ) -> Optional[str]:
        """
        Get template for trigger and movement state.

        Args:
            trigger_id: Trigger identifier
            movement_state: Movement state ("combat" or "non_combat")
            **kwargs: Template variables

        Returns:
            Template string or None if not found
        """
        template_key = f"{trigger_id}_{movement_state}"
        template = self.templates.get(template_key)

        if template is None:
            return None

        # Replace template variables
        if kwargs:
            template = template.format(**kwargs)

        return template

    def add_template(self, key: str, template: str) -> None:
        """
        Add a new template.

        Args:
            key: Template key
            template: Template string
        """
        self.templates[key] = template

    def set_templates_from_dict(self, templates: Dict[str, str]) -> None:
        """
        Set templates from dictionary.

        Args:
            templates: Dictionary of template keys to template strings
        """
        self.templates.update(templates)

    def get_short_template(
        self,
        trigger_id: str,
        movement_state: str,
        **kwargs
    ) -> Optional[str]:
        """
        Get short template (max 2 sentences).

        Args:
            trigger_id: Trigger identifier
            movement_state: Movement state
            **kwargs: Template variables

        Returns:
            Short template string or None
        """
        template = self.get_template(trigger_id, movement_state, **kwargs)

        if template is None:
            return None

        # Truncate to 2 sentences
        sentences = template.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        return template
