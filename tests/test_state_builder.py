"""Tests for StateBuilder."""

import pytest
from vision.state_builder import StateBuilder


class TestStateBuilder:
    """Test StateBuilder functionality."""

    def test_init(self):
        """Test StateBuilder initialization."""
        builder = StateBuilder()

        assert builder.current_state is not None
        assert "player" in builder.current_state
        assert "world" in builder.current_state
        assert "session" in builder.current_state

    def test_create_empty_state_structure(self):
        """Test empty state has correct structure."""
        builder = StateBuilder()
        state = builder.current_state

        # Check player structure
        assert "player" in state
        assert "status" in state["player"]
        assert "weapon" in state["player"]
        assert "inventory" in state["player"]

        # Check player.status fields
        assert "hp" in state["player"]["status"]
        assert "shield" in state["player"]["status"]
        assert "is_knocked" in state["player"]["status"]

        # Check player.weapon fields
        assert "name" in state["player"]["weapon"]
        assert "ammo" in state["player"]["weapon"]

        # Check player.inventory fields
        assert "materials" in state["player"]["inventory"]

        # Check world structure
        assert "world" in state
        assert "storm" in state["world"]

        # Check world.storm fields
        assert "phase" in state["world"]["storm"]
        assert "damage" in state["world"]["storm"]
        assert "in_storm" in state["world"]["storm"]
        assert "is_shrinking" in state["world"]["storm"]
        assert "next_circle_distance" in state["world"]["storm"]

        # Check session structure
        assert "session" in state
        assert "phase" in state["session"]
        assert "inactivity_duration_ms" in state["session"]

    def test_create_empty_state_default_values(self):
        """Test empty state has correct default values."""
        builder = StateBuilder()
        state = builder.current_state

        # Check HP defaults
        hp_field = state["player"]["status"]["hp"]
        assert hp_field["value"] == 100
        assert hp_field["source"] == "default"
        assert hp_field["confidence"] == 1.0
        assert "ts_ms" in hp_field

        # Check shield defaults
        shield_field = state["player"]["status"]["shield"]
        assert shield_field["value"] == 0

        # Check is_knocked defaults
        knocked_field = state["player"]["status"]["is_knocked"]
        assert knocked_field["value"] is False

        # Check weapon name defaults (should be None with 0 confidence)
        weapon_name_field = state["player"]["weapon"]["name"]
        assert weapon_name_field["value"] is None
        assert weapon_name_field["confidence"] == 0.0

        # Check in_storm defaults
        in_storm_field = state["world"]["storm"]["in_storm"]
        assert in_storm_field["value"] is False

    def test_update_field(self):
        """Test updating a specific field."""
        builder = StateBuilder()
        builder.update_field("player.status.hp", 75, "test_source", 0.95)

        hp_field = builder.current_state["player"]["status"]["hp"]
        assert hp_field["value"] == 75
        assert hp_field["source"] == "test_source"
        assert hp_field["confidence"] == 0.95

    def test_update_field_nested_dict_structure(self):
        """Test updating a field maintains the nested dict structure."""
        builder = StateBuilder()
        builder.update_field("player.status.hp", 50, "roi", 0.9)

        hp_field = builder.current_state["player"]["status"]["hp"]
        assert isinstance(hp_field, dict)
        assert "value" in hp_field
        assert "source" in hp_field
        assert "confidence" in hp_field
        assert "ts_ms" in hp_field

    def test_update_field_invalid_path(self):
        """Test updating with invalid path does nothing."""
        builder = StateBuilder()
        original_hp = builder.current_state["player"]["status"]["hp"]["value"]

        # This should not crash or change anything
        builder.update_field("invalid.path.to.field", 100, "test", 1.0)

        assert builder.current_state["player"]["status"]["hp"]["value"] == original_hp

    def test_update_hp(self):
        """Test update_hp convenience method."""
        builder = StateBuilder()
        builder.update_hp(80, "roi", 0.85)

        hp_field = builder.current_state["player"]["status"]["hp"]
        assert hp_field["value"] == 80
        assert hp_field["source"] == "roi"
        assert hp_field["confidence"] == 0.85

    def test_update_shield(self):
        """Test update_shield convenience method."""
        builder = StateBuilder()
        builder.update_shield(50, "roi", 0.9)

        shield_field = builder.current_state["player"]["status"]["shield"]
        assert shield_field["value"] == 50
        assert shield_field["source"] == "roi"

    def test_update_knocked(self):
        """Test update_knocked convenience method."""
        builder = StateBuilder()
        builder.update_knocked(True, "roi", 1.0)

        knocked_field = builder.current_state["player"]["status"]["is_knocked"]
        assert knocked_field["value"] is True
        assert knocked_field["confidence"] == 1.0

    def test_update_weapon_name(self):
        """Test update_weapon_name convenience method."""
        builder = StateBuilder()
        builder.update_weapon_name("Assault Rifle", "yolo", 0.92)

        weapon_field = builder.current_state["player"]["weapon"]["name"]
        assert weapon_field["value"] == "Assault Rifle"
        assert weapon_field["source"] == "yolo"
        assert weapon_field["confidence"] == 0.92

    def test_update_ammo(self):
        """Test update_ammo convenience method."""
        builder = StateBuilder()
        builder.update_ammo(30, "ocr", 0.95)

        ammo_field = builder.current_state["player"]["weapon"]["ammo"]
        assert ammo_field["value"] == 30
        assert ammo_field["source"] == "ocr"

    def test_update_materials(self):
        """Test update_materials convenience method."""
        builder = StateBuilder()
        builder.update_materials(200, "ocr", 0.98)

        materials_field = builder.current_state["player"]["inventory"]["materials"]
        assert materials_field["value"] == 200
        assert materials_field["source"] == "ocr"

    def test_update_storm_phase(self):
        """Test update_storm_phase convenience method."""
        builder = StateBuilder()
        builder.update_storm_phase(3, "roi", 1.0)

        phase_field = builder.current_state["world"]["storm"]["phase"]
        assert phase_field["value"] == 3
        assert phase_field["source"] == "roi"

    def test_update_storm_damage(self):
        """Test update_storm_damage convenience method."""
        builder = StateBuilder()
        builder.update_storm_damage(5.5, "roi", 1.0)

        damage_field = builder.current_state["world"]["storm"]["damage"]
        assert damage_field["value"] == 5.5

    def test_update_in_storm(self):
        """Test update_in_storm convenience method."""
        builder = StateBuilder()
        builder.update_in_storm(True, "roi", 1.0)

        in_storm_field = builder.current_state["world"]["storm"]["in_storm"]
        assert in_storm_field["value"] is True

    def test_update_storm_shrinking(self):
        """Test update_storm_shrinking convenience method."""
        builder = StateBuilder()
        builder.update_storm_shrinking(True, "roi", 1.0)

        shrinking_field = builder.current_state["world"]["storm"]["is_shrinking"]
        assert shrinking_field["value"] is True

    def test_update_session_phase(self):
        """Test update_session_phase convenience method."""
        builder = StateBuilder()
        builder.update_session_phase("combat", "system", 1.0)

        phase_field = builder.current_state["session"]["phase"]
        assert phase_field["value"] == "combat"
        assert phase_field["source"] == "system"

    def test_get_state(self):
        """Test getting current state."""
        builder = StateBuilder()
        builder.update_hp(75, "test", 0.9)

        state = builder.get_state()
        assert state is builder.current_state
        assert state["player"]["status"]["hp"]["value"] == 75

    def test_reset(self):
        """Test resetting state to empty template."""
        builder = StateBuilder()
        builder.update_hp(25, "test", 0.9)
        builder.update_shield(0, "test", 1.0)
        builder.update_weapon_name("Shotgun", "test", 0.9)

        builder.reset()

        # After reset, should be back to defaults
        assert builder.current_state["player"]["status"]["hp"]["value"] == 100
        assert builder.current_state["player"]["status"]["shield"]["value"] == 0
        assert builder.current_state["player"]["weapon"]["name"]["value"] is None

    def test_get_movement_state_combat_low_hp(self):
        """Test movement state is combat when HP is low."""
        builder = StateBuilder()
        builder.update_hp(40, "test", 1.0)

        assert builder.get_movement_state() == "combat"

    def test_get_movement_state_combat_in_storm(self):
        """Test movement state is combat when in storm."""
        builder = StateBuilder()
        builder.update_hp(100, "test", 1.0)  # High HP
        builder.update_in_storm(True, "test", 1.0)

        assert builder.get_movement_state() == "combat"

    def test_get_movement_state_non_combat(self):
        """Test movement state is non_combat when safe."""
        builder = StateBuilder()
        builder.update_hp(100, "test", 1.0)
        builder.update_in_storm(False, "test", 1.0)

        assert builder.get_movement_state() == "non_combat"

    def test_get_movement_state_hp_threshold_50(self):
        """Test HP threshold is exactly 50."""
        builder = StateBuilder()

        # HP = 50 should be non_combat (not less than 50)
        builder.update_hp(50, "test", 1.0)
        assert builder.get_movement_state() == "non_combat"

        # HP = 49 should be combat
        builder.update_hp(49, "test", 1.0)
        assert builder.get_movement_state() == "combat"

    def test_multiple_updates(self):
        """Test multiple field updates."""
        builder = StateBuilder()

        builder.update_hp(60, "roi", 0.9)
        builder.update_shield(25, "roi", 0.9)
        builder.update_weapon_name("Pistol", "yolo", 0.85)
        builder.update_ammo(15, "ocr", 0.95)
        builder.update_materials(150, "ocr", 0.98)
        builder.update_in_storm(True, "roi", 1.0)

        state = builder.get_state()
        assert state["player"]["status"]["hp"]["value"] == 60
        assert state["player"]["status"]["shield"]["value"] == 25
        assert state["player"]["weapon"]["name"]["value"] == "Pistol"
        assert state["player"]["weapon"]["ammo"]["value"] == 15
        assert state["player"]["inventory"]["materials"]["value"] == 150
        assert state["world"]["storm"]["in_storm"]["value"] is True

    def test_timestamp_updates(self):
        """Test that timestamps are updated on field update."""
        builder = StateBuilder()
        builder.update_hp(75, "test", 0.9)

        ts_ms = builder.current_state["player"]["status"]["hp"]["ts_ms"]
        assert ts_ms > 0
        assert isinstance(ts_ms, int)

    def test_state_immutability_for_external_modification(self):
        """Test that external modifications don't affect internal state unexpectedly."""
        builder = StateBuilder()
        state = builder.get_state()

        # Modify the returned state
        state["player"]["status"]["hp"]["value"] = 999

        # The internal state should also be affected (since we return reference)
        # This is expected behavior - just documenting it
        assert builder.current_state["player"]["status"]["hp"]["value"] == 999
