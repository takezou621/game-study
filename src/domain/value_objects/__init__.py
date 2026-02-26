"""Value objects - immutable objects defined by their attributes."""

from domain.value_objects.ammo import Ammo
from domain.value_objects.health import HP, Shield
from domain.value_objects.position import Position

__all__ = [
    "HP",
    "Shield",
    "Ammo",
    "Position",
]
