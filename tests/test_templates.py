"""Tests for DialogueTemplateManager."""

import pytest

from dialogue.templates import DialogueTemplateManager


class TestDialogueTemplateManager:
    """Test DialogueTemplateManager functionality."""

    def test_init(self):
        """Test DialogueTemplateManager initialization."""
        manager = DialogueTemplateManager()

        assert manager.templates is not None
        assert isinstance(manager.templates, dict)
        assert len(manager.templates) > 0

    def test_default_templates_exist(self):
        """Test that default templates are loaded."""
        manager = DialogueTemplateManager()

        # Check P0 templates exist
        assert "p0_knocked_combat" in manager.templates
        assert "p0_knocked_non_combat" in manager.templates
        assert "p0_low_hp_combat" in manager.templates
        assert "p0_low_hp_non_combat" in manager.templates
        assert "p0_storm_damage_combat" in manager.templates
        assert "p0_storm_damage_non_combat" in manager.templates

        # Check P1 templates exist
        assert "p1_storm_shrinking_combat" in manager.templates
        assert "p1_storm_shrinking_non_combat" in manager.templates

        # Check P2 templates exist
        assert "p2_weapon_non_combat" in manager.templates

        # Check P3 templates exist
        assert "p3_session_summary" in manager.templates
        assert "p3_small_talk" in manager.templates

    def test_get_template_exists(self):
        """Test getting an existing template."""
        manager = DialogueTemplateManager()

        template = manager.get_template("p0_knocked", "combat")

        assert template is not None
        assert "You're knocked" in template

    def test_get_template_with_combat_state(self):
        """Test getting template for combat state."""
        manager = DialogueTemplateManager()

        template = manager.get_template("p0_low_hp", "combat")

        assert template == "Low HP! Find cover immediately!"

    def test_get_template_with_non_combat_state(self):
        """Test getting template for non-combat state."""
        manager = DialogueTemplateManager()

        template = manager.get_template("p0_low_hp", "non_combat")

        assert template == "Your health is critical. Heal up before fighting."

    def test_get_template_not_found(self):
        """Test getting a non-existent template."""
        manager = DialogueTemplateManager()

        template = manager.get_template("non_existent_trigger", "combat")

        assert template is None

    def test_get_template_with_state_not_found(self):
        """Test getting template when state variant doesn't exist."""
        manager = DialogueTemplateManager()

        # p3_session_summary only has non_combat variant
        template = manager.get_template("p3_session_summary", "combat")

        assert template is None

    def test_get_template_with_kwargs(self):
        """Test getting template with variable substitution."""
        manager = DialogueTemplateManager()

        template = manager.get_template(
            "p2_weapon",
            "non_combat",
            weapon_name="Assault Rifle",
            situation="medium range combat"
        )

        assert template is not None
        assert "Assault Rifle" in template
        assert "medium range combat" in template

    def test_get_template_with_partial_kwargs(self):
        """Test getting template with partial variable substitution."""
        manager = DialogueTemplateManager()

        # This should raise KeyError due to missing format variables
        with pytest.raises(KeyError):
            manager.get_template("p2_weapon", "non_combat", weapon_name="Rifle")

    def test_get_template_with_extra_kwargs(self):
        """Test getting template with extra kwargs (should be ignored)."""
        manager = DialogueTemplateManager()

        template = manager.get_template(
            "p0_knocked",
            "combat",
            extra_var="should be ignored",
            another_var="also ignored"
        )

        # Template without format variables should work fine
        assert template is not None
        assert "You're knocked" in template

    def test_add_template(self):
        """Test adding a new template."""
        manager = DialogueTemplateManager()

        assert "custom_trigger_combat" not in manager.templates

        manager.add_template("custom_trigger_combat", "Custom combat message")

        assert "custom_trigger_combat" in manager.templates
        assert manager.templates["custom_trigger_combat"] == "Custom combat message"

    def test_add_template_overwrite(self):
        """Test that add_template overwrites existing template."""
        manager = DialogueTemplateManager()

        original = manager.templates["p0_knocked_combat"]
        assert "You're knocked" in original

        manager.add_template("p0_knocked_combat", "New knocked message")

        assert manager.templates["p0_knocked_combat"] == "New knocked message"

    def test_set_templates_from_dict(self):
        """Test setting templates from dictionary."""
        manager = DialogueTemplateManager()

        new_templates = {
            "new_template_1": "Message 1",
            "new_template_2": "Message 2",
        }

        manager.set_templates_from_dict(new_templates)

        assert "new_template_1" in manager.templates
        assert "new_template_2" in manager.templates
        assert manager.templates["new_template_1"] == "Message 1"

    def test_set_templates_from_dict_merges(self):
        """Test that set_templates_from_dict merges with existing."""
        manager = DialogueTemplateManager()

        original_count = len(manager.templates)

        new_templates = {
            "new_template": "New message",
        }

        manager.set_templates_from_dict(new_templates)

        # Should have original + new
        assert len(manager.templates) == original_count + 1
        assert "p0_knocked_combat" in manager.templates  # Original still there
        assert "new_template" in manager.templates  # New added

    def test_get_short_template_single_sentence(self):
        """Test getting short template with single sentence."""
        manager = DialogueTemplateManager()

        template = manager.get_short_template("p0_knocked", "combat")

        # Single sentence should be returned as is
        assert template is not None
        assert "You're knocked" in template

    def test_get_short_template_two_sentences(self):
        """Test getting short template with two sentences."""
        manager = DialogueTemplateManager()

        template = manager.get_short_template("p0_low_hp", "non_combat")

        # Two sentences should be returned as is
        assert template is not None
        assert template.count('.') >= 1

    def test_get_short_template_truncates(self):
        """Test that get_short_template truncates to 2 sentences."""
        manager = DialogueTemplateManager()

        # Add a template with more than 2 sentences
        manager.add_template(
            "long_template_combat",
            "First sentence. Second sentence. Third sentence. Fourth sentence."
        )

        template = manager.get_short_template("long_template", "combat")

        assert template is not None
        sentences = template.split('. ')
        assert len(sentences) == 2
        assert template.endswith('.')

    def test_get_short_template_not_found(self):
        """Test get_short_template with non-existent template."""
        manager = DialogueTemplateManager()

        template = manager.get_short_template("non_existent", "combat")

        assert template is None

    def test_template_format_with_braces(self):
        """Test template with literal braces (format escaping)."""
        manager = DialogueTemplateManager()

        # Add a template with braces that should be preserved
        manager.add_template("braces_test_combat", "Value: {value} and literal {{braces}}")

        template = manager.get_template("braces_test", "combat", value=42)

        assert "Value: 42" in template

    def test_template_empty_string(self):
        """Test getting empty template."""
        manager = DialogueTemplateManager()

        manager.add_template("empty_template_combat", "")

        template = manager.get_template("empty_template", "combat")

        assert template == ""

    def test_template_with_newlines(self):
        """Test template with newline characters."""
        manager = DialogueTemplateManager()

        manager.add_template("newline_template_combat", "Line 1\nLine 2\nLine 3")

        template = manager.get_template("newline_template", "combat")

        assert "\n" in template
        assert template.count("\n") == 2

    def test_get_template_preserves_original(self):
        """Test that getting template doesn't modify the original."""
        manager = DialogueTemplateManager()

        original_template = manager.templates["p2_weapon_non_combat"]

        # Get template with substitutions
        template = manager.get_template(
            "p2_weapon",
            "non_combat",
            weapon_name="Shotgun",
            situation="close range"
        )

        # Original should still have placeholders
        assert "{weapon_name}" in original_template
        assert "{situation}" in original_template

        # Returned template should have substitutions
        assert "Shotgun" in template
        assert "close range" in template

    def test_multiple_template_managers(self):
        """Test that multiple managers have independent templates."""
        manager1 = DialogueTemplateManager()
        manager2 = DialogueTemplateManager()

        manager1.add_template("custom", "Manager 1 template")

        assert "custom" in manager1.templates
        assert "custom" not in manager2.templates

        manager2.add_template("custom", "Manager 2 template")

        assert manager1.templates["custom"] == "Manager 1 template"
        assert manager2.templates["custom"] == "Manager 2 template"

    def test_template_priority_structure(self):
        """Test that templates follow priority naming convention."""
        manager = DialogueTemplateManager()

        # Check P0 (Survival) templates
        p0_keys = [k for k in manager.templates.keys() if k.startswith("p0_")]
        assert len(p0_keys) >= 3

        # Check P1 (Tactical) templates
        p1_keys = [k for k in manager.templates.keys() if k.startswith("p1_")]
        assert len(p1_keys) >= 2

        # Check P2 (Learning) templates
        p2_keys = [k for k in manager.templates.keys() if k.startswith("p2_")]
        assert len(p2_keys) >= 1

        # Check P3 (Chatter) templates
        p3_keys = [k for k in manager.templates.keys() if k.startswith("p3_")]
        assert len(p3_keys) >= 1
