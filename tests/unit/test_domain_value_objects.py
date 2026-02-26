"""Tests for domain value objects: HP, Shield, Ammo, Position."""

import importlib.util
from pathlib import Path

import pytest

# Direct module imports to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"

# Load exceptions first (needed by value objects)
exceptions_spec = importlib.util.spec_from_file_location(
    "domain_exceptions",
    SRC_PATH / "domain" / "exceptions.py",
)
exceptions_module = importlib.util.module_from_spec(exceptions_spec)
exceptions_spec.loader.exec_module(exceptions_module)

# Make domain.exceptions available for import
import sys

sys.modules["domain.exceptions"] = exceptions_module

# Load value objects
health_spec = importlib.util.spec_from_file_location(
    "health",
    SRC_PATH / "domain" / "value_objects" / "health.py",
)
health_module = importlib.util.module_from_spec(health_spec)
health_spec.loader.exec_module(health_module)

ammo_spec = importlib.util.spec_from_file_location(
    "ammo",
    SRC_PATH / "domain" / "value_objects" / "ammo.py",
)
ammo_module = importlib.util.module_from_spec(ammo_spec)
ammo_spec.loader.exec_module(ammo_module)

position_spec = importlib.util.spec_from_file_location(
    "position",
    SRC_PATH / "domain" / "value_objects" / "position.py",
)
position_module = importlib.util.module_from_spec(position_spec)
position_spec.loader.exec_module(position_module)

# Import classes
HP = health_module.HP
Shield = health_module.Shield
Ammo = ammo_module.Ammo
Position = position_module.Position
InvalidValueError = exceptions_module.InvalidValueError


class TestHP:
    """Tests for HP value object."""

    def test_init_with_valid_value(self):
        """Test HP initialization with valid value."""
        hp = HP(value=75)
        assert hp.value == 75
        assert hp.max_value == 100

    def test_init_with_custom_max(self):
        """Test HP initialization with custom max_value."""
        hp = HP(value=150, max_value=200)
        assert hp.value == 150
        assert hp.max_value == 200

    def test_init_at_boundary_min(self):
        """Test HP initialization at minimum boundary."""
        hp = HP(value=0)
        assert hp.value == 0

    def test_init_at_boundary_max(self):
        """Test HP initialization at maximum boundary."""
        hp = HP(value=100)
        assert hp.value == 100

    def test_init_negative_raises_error(self):
        """Test HP initialization with negative value raises error."""
        with pytest.raises(InvalidValueError, match="HP must be between 0 and"):
            HP(value=-1)

    def test_init_exceeds_max_raises_error(self):
        """Test HP initialization exceeding max raises error."""
        with pytest.raises(InvalidValueError, match="HP must be between 0 and"):
            HP(value=101)

    def test_is_low_true(self):
        """Test is_low returns True when HP < 50."""
        hp = HP(value=49)
        assert hp.is_low is True

    def test_is_low_false(self):
        """Test is_low returns False when HP >= 50."""
        hp = HP(value=50)
        assert hp.is_low is False

    def test_is_critical_true(self):
        """Test is_critical returns True when HP < 25."""
        hp = HP(value=24)
        assert hp.is_critical is True

    def test_is_critical_false(self):
        """Test is_critical returns False when HP >= 25."""
        hp = HP(value=25)
        assert hp.is_critical is False

    def test_percentage(self):
        """Test percentage calculation."""
        hp = HP(value=75)
        assert hp.percentage == 0.75

    def test_percentage_with_custom_max(self):
        """Test percentage with custom max_value."""
        hp = HP(value=150, max_value=200)
        assert hp.percentage == 0.75

    def test_with_value(self):
        """Test with_value creates new HP instance."""
        hp = HP(value=75)
        new_hp = hp.with_value(50)
        assert new_hp.value == 50
        assert hp.value == 75  # Original unchanged
        assert new_hp.max_value == hp.max_value

    def test_to_dict(self):
        """Test to_dict method."""
        hp = HP(value=75)
        result = hp.to_dict()
        assert result == {"value": 75, "max_value": 100}

    def test_frozen(self):
        """Test HP is immutable (frozen dataclass)."""
        hp = HP(value=75)
        with pytest.raises(AttributeError):
            hp.value = 50  # type: ignore

    def test_equality(self):
        """Test HP equality comparison."""
        hp1 = HP(value=75)
        hp2 = HP(value=75)
        hp3 = HP(value=50)
        assert hp1 == hp2
        assert hp1 != hp3


class TestShield:
    """Tests for Shield value object."""

    def test_init_with_valid_value(self):
        """Test Shield initialization with valid value."""
        shield = Shield(value=50)
        assert shield.value == 50
        assert shield.max_value == 100

    def test_init_with_custom_max(self):
        """Test Shield initialization with custom max_value."""
        shield = Shield(value=25, max_value=50)
        assert shield.value == 25
        assert shield.max_value == 50

    def test_init_at_boundary_min(self):
        """Test Shield initialization at minimum boundary."""
        shield = Shield(value=0)
        assert shield.value == 0

    def test_init_at_boundary_max(self):
        """Test Shield initialization at maximum boundary."""
        shield = Shield(value=100)
        assert shield.value == 100

    def test_init_negative_raises_error(self):
        """Test Shield initialization with negative value raises error."""
        with pytest.raises(InvalidValueError, match="Shield must be between 0 and"):
            Shield(value=-10)

    def test_init_exceeds_max_raises_error(self):
        """Test Shield initialization exceeding max raises error."""
        with pytest.raises(InvalidValueError, match="Shield must be between 0 and"):
            Shield(value=101)

    def test_is_active_true(self):
        """Test is_active returns True when shield > 0."""
        shield = Shield(value=1)
        assert shield.is_active is True

    def test_is_active_false(self):
        """Test is_active returns False when shield = 0."""
        shield = Shield(value=0)
        assert shield.is_active is False

    def test_percentage(self):
        """Test percentage calculation."""
        shield = Shield(value=50)
        assert shield.percentage == 0.5

    def test_with_value(self):
        """Test with_value creates new Shield instance."""
        shield = Shield(value=50)
        new_shield = shield.with_value(75)
        assert new_shield.value == 75
        assert shield.value == 50  # Original unchanged

    def test_to_dict(self):
        """Test to_dict method."""
        shield = Shield(value=75, max_value=100)
        result = shield.to_dict()
        assert result == {"value": 75, "max_value": 100}

    def test_frozen(self):
        """Test Shield is immutable (frozen dataclass)."""
        shield = Shield(value=50)
        with pytest.raises(AttributeError):
            shield.value = 25  # type: ignore


class TestAmmo:
    """Tests for Ammo value object."""

    def test_init_with_valid_value(self):
        """Test Ammo initialization with valid value."""
        ammo = Ammo(value=30)
        assert ammo.value == 30
        assert ammo.max_value is None

    def test_init_with_max_value(self):
        """Test Ammo initialization with max_value."""
        ammo = Ammo(value=20, max_value=30)
        assert ammo.value == 20
        assert ammo.max_value == 30

    def test_init_zero(self):
        """Test Ammo initialization with zero."""
        ammo = Ammo(value=0)
        assert ammo.value == 0

    def test_init_negative_raises_error(self):
        """Test Ammo initialization with negative value raises error."""
        with pytest.raises(InvalidValueError, match="Ammo cannot be negative"):
            Ammo(value=-1)

    def test_init_exceeds_max_raises_error(self):
        """Test Ammo initialization exceeding max raises error."""
        with pytest.raises(InvalidValueError, match="Ammo cannot exceed max"):
            Ammo(value=35, max_value=30)

    def test_is_empty_true(self):
        """Test is_empty returns True when ammo = 0."""
        ammo = Ammo(value=0)
        assert ammo.is_empty is True

    def test_is_empty_false(self):
        """Test is_empty returns False when ammo > 0."""
        ammo = Ammo(value=1)
        assert ammo.is_empty is False

    def test_is_low_without_max(self):
        """Test is_low returns True when ammo < 10 (no max)."""
        ammo = Ammo(value=9)
        assert ammo.is_low is True

        ammo = Ammo(value=10)
        assert ammo.is_low is False

    def test_is_low_with_max(self):
        """Test is_low returns True when ammo < 20% of max."""
        ammo = Ammo(value=5, max_value=30)
        assert ammo.is_low is True  # 5 < 30 * 0.2 = 6

        ammo = Ammo(value=6, max_value=30)
        assert ammo.is_low is False  # 6 >= 6

    def test_percentage_without_max(self):
        """Test percentage returns None when max_value is None."""
        ammo = Ammo(value=30)
        assert ammo.percentage is None

    def test_percentage_with_max(self):
        """Test percentage calculation with max_value."""
        ammo = Ammo(value=15, max_value=30)
        assert ammo.percentage == 0.5

    def test_with_value(self):
        """Test with_value creates new Ammo instance."""
        ammo = Ammo(value=20, max_value=30)
        new_ammo = ammo.with_value(10)
        assert new_ammo.value == 10
        assert ammo.value == 20  # Original unchanged
        assert new_ammo.max_value == ammo.max_value

    def test_to_dict(self):
        """Test to_dict method."""
        ammo = Ammo(value=20, max_value=30)
        result = ammo.to_dict()
        assert result == {"value": 20, "max_value": 30}

    def test_to_dict_without_max(self):
        """Test to_dict method without max_value."""
        ammo = Ammo(value=20)
        result = ammo.to_dict()
        assert result == {"value": 20, "max_value": None}

    def test_frozen(self):
        """Test Ammo is immutable (frozen dataclass)."""
        ammo = Ammo(value=30)
        with pytest.raises(AttributeError):
            ammo.value = 20  # type: ignore


class TestPosition:
    """Tests for Position value object."""

    def test_init_with_valid_coordinates(self):
        """Test Position initialization with valid coordinates."""
        pos = Position(x=0.5, y=0.5)
        assert pos.x == 0.5
        assert pos.y == 0.5

    def test_init_with_pixel_coordinates(self):
        """Test Position supports pixel coordinates."""
        pos = Position(x=1920, y=1080)
        assert pos.x == 1920
        assert pos.y == 1080

    def test_init_with_negative_coordinates(self):
        """Test Position allows negative coordinates."""
        pos = Position(x=-10, y=-20)
        assert pos.x == -10
        assert pos.y == -20

    def test_is_normalized_true(self):
        """Test is_normalized returns True for normalized coordinates."""
        pos = Position(x=0.5, y=0.5)
        assert pos.is_normalized is True

    def test_is_normalized_false(self):
        """Test is_normalized returns False for non-normalized coordinates."""
        pos = Position(x=1.5, y=0.5)
        assert pos.is_normalized is False

        pos = Position(x=1920, y=1080)
        assert pos.is_normalized is False

    def test_is_normalized_at_boundary(self):
        """Test is_normalized at boundaries (0 and 1)."""
        pos = Position(x=0, y=1)
        assert pos.is_normalized is True

    def test_distance_to_same_point(self):
        """Test distance_to returns 0 for same point."""
        pos = Position(x=0.5, y=0.5)
        assert pos.distance_to(Position(x=0.5, y=0.5)) == 0

    def test_distance_to_horizontal(self):
        """Test distance_to for horizontal distance."""
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=3, y=0)
        assert pos1.distance_to(pos2) == 3

    def test_distance_to_vertical(self):
        """Test distance_to for vertical distance."""
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=0, y=4)
        assert pos1.distance_to(pos2) == 4

    def test_distance_to_diagonal(self):
        """Test distance_to for diagonal distance (3-4-5 triangle)."""
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=3, y=4)
        assert pos1.distance_to(pos2) == 5

    def test_distance_to_symmetric(self):
        """Test distance_to is symmetric."""
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=3, y=4)
        assert pos1.distance_to(pos2) == pos2.distance_to(pos1)

    def test_to_dict(self):
        """Test to_dict method."""
        pos = Position(x=0.5, y=0.75)
        result = pos.to_dict()
        assert result == {"x": 0.5, "y": 0.75}

    def test_to_tuple(self):
        """Test to_tuple method."""
        pos = Position(x=0.5, y=0.75)
        result = pos.to_tuple()
        assert result == (0.5, 0.75)
        assert isinstance(result, tuple)

    def test_frozen(self):
        """Test Position is immutable (frozen dataclass)."""
        pos = Position(x=0.5, y=0.5)
        with pytest.raises(AttributeError):
            pos.x = 0.6  # type: ignore

    def test_equality(self):
        """Test Position equality comparison."""
        pos1 = Position(x=0.5, y=0.5)
        pos2 = Position(x=0.5, y=0.5)
        pos3 = Position(x=0.5, y=0.6)
        assert pos1 == pos2
        assert pos1 != pos3


class TestValueObjectIntegration:
    """Integration tests for value objects working together."""

    def test_hp_and_shield_independence(self):
        """Test HP and Shield are independent."""
        hp = HP(value=50)
        shield = Shield(value=75)
        assert hp.value != shield.value
        assert hp.max_value == shield.max_value  # Same max

    def test_ammo_with_different_max_values(self):
        """Test Ammo instances with different max values."""
        pistol_ammo = Ammo(value=12, max_value=12)
        rifle_ammo = Ammo(value=30, max_value=30)
        assert pistol_ammo.is_empty is False
        assert rifle_ammo.is_empty is False
        assert pistol_ammo.max_value != rifle_ammo.max_value

    def test_position_calculations_with_different_scales(self):
        """Test Position works with different coordinate scales."""
        normalized = Position(x=0.5, y=0.5)
        pixel = Position(x=960, y=540)
        # They can be used independently
        assert normalized.is_normalized is True
        assert pixel.is_normalized is False
