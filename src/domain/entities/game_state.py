"""Game state entity - the aggregate root for game state."""

from dataclasses import dataclass, field
from typing import Any

from domain.entities.player import PlayerStatus
from domain.entities.session import Session
from domain.value_objects.ammo import Ammo
from domain.value_objects.health import HP, Shield
from utils.time import get_timestamp_ms

# Movement state constants
MOVEMENT_STATE_COMBAT = "combat"
MOVEMENT_STATE_NON_COMBAT = "non_combat"
MOVEMENT_STATES = [MOVEMENT_STATE_COMBAT, MOVEMENT_STATE_NON_COMBAT]


@dataclass
class WeaponInfo:
    """Weapon information."""

    name: str | None = None
    ammo: Ammo = field(default_factory=lambda: Ammo(value=0))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "ammo": self.ammo.to_dict() if self.ammo else None,
        }


@dataclass
class InventoryInfo:
    """Inventory information."""

    materials: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "materials": self.materials,
        }


@dataclass
class StormInfo:
    """Storm information."""

    phase: int | None = None
    damage: float | None = None
    in_storm: bool = False
    is_shrinking: bool = False
    next_circle_distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase,
            "damage": self.damage,
            "in_storm": self.in_storm,
            "is_shrinking": self.is_shrinking,
            "next_circle_distance": self.next_circle_distance,
        }


@dataclass
class WorldInfo:
    """World information."""

    storm: StormInfo = field(default_factory=StormInfo)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storm": self.storm.to_dict(),
        }


@dataclass
class Player:
    """Player aggregate."""

    status: PlayerStatus = field(default_factory=PlayerStatus)
    weapon: WeaponInfo = field(default_factory=WeaponInfo)
    inventory: InventoryInfo = field(default_factory=InventoryInfo)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.to_dict(),
            "weapon": self.weapon.to_dict(),
            "inventory": self.inventory.to_dict(),
        }


@dataclass
class GameState:
    """Game state aggregate root.

    This is the central entity that represents the complete game state,
    including player status, world information, and session data.
    """

    player: Player = field(default_factory=Player)
    world: WorldInfo = field(default_factory=WorldInfo)
    session: Session = field(default_factory=Session)
    updated_at_ms: int = field(default_factory=get_timestamp_ms)

    @property
    def is_combat(self) -> bool:
        """Determine if player is in combat situation."""
        # Heuristic: if HP < 50 or in storm, consider combat
        hp = self.player.status.hp.value
        in_storm = self.world.storm.in_storm
        return (hp is not None and hp < 50) or in_storm

    @property
    def movement_state(self) -> str:
        """Get current movement state."""
        return MOVEMENT_STATE_COMBAT if self.is_combat else MOVEMENT_STATE_NON_COMBAT

    @property
    def needs_attention(self) -> bool:
        """Check if player needs attention (low HP, in storm, etc.)."""
        return (
            self.player.status.hp.is_low
            or self.world.storm.in_storm
            or self.player.status.is_knocked
        )

    def with_player_status(self, status: PlayerStatus) -> "GameState":
        """Create new GameState with updated player status."""
        return GameState(
            player=Player(
                status=status,
                weapon=self.player.weapon,
                inventory=self.player.inventory,
            ),
            world=self.world,
            session=self.session,
            updated_at_ms=get_timestamp_ms(),
        )

    def with_hp(self, hp: int) -> "GameState":
        """Create new GameState with updated HP."""
        new_status = PlayerStatus(
            hp=HP(value=hp),
            shield=self.player.status.shield,
            is_knocked=self.player.status.is_knocked,
        )
        return self.with_player_status(new_status)

    def with_shield(self, shield: int) -> "GameState":
        """Create new GameState with updated shield."""
        new_status = PlayerStatus(
            hp=self.player.status.hp,
            shield=Shield(value=shield),
            is_knocked=self.player.status.is_knocked,
        )
        return self.with_player_status(new_status)

    def with_knocked(self, is_knocked: bool) -> "GameState":
        """Create new GameState with knocked status."""
        new_status = PlayerStatus(
            hp=self.player.status.hp,
            shield=self.player.status.shield,
            is_knocked=is_knocked,
        )
        return self.with_player_status(new_status)

    def with_weapon(self, name: str | None, ammo: int | None) -> "GameState":
        """Create new GameState with weapon info."""
        return GameState(
            player=Player(
                status=self.player.status,
                weapon=WeaponInfo(
                    name=name,
                    ammo=Ammo(value=ammo) if ammo is not None else Ammo(value=0),
                ),
                inventory=self.player.inventory,
            ),
            world=self.world,
            session=self.session,
            updated_at_ms=get_timestamp_ms(),
        )

    def with_materials(self, materials: int) -> "GameState":
        """Create new GameState with materials count."""
        return GameState(
            player=Player(
                status=self.player.status,
                weapon=self.player.weapon,
                inventory=InventoryInfo(materials=materials),
            ),
            world=self.world,
            session=self.session,
            updated_at_ms=get_timestamp_ms(),
        )

    def with_storm(
        self,
        phase: int | None = None,
        damage: float | None = None,
        in_storm: bool = False,
        is_shrinking: bool = False,
        next_circle_distance: float | None = None,
    ) -> "GameState":
        """Create new GameState with storm info."""
        return GameState(
            player=self.player,
            world=WorldInfo(
                storm=StormInfo(
                    phase=phase,
                    damage=damage,
                    in_storm=in_storm,
                    is_shrinking=is_shrinking,
                    next_circle_distance=next_circle_distance,
                )
            ),
            session=self.session,
            updated_at_ms=get_timestamp_ms(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trigger evaluation and API responses."""
        return {
            "player": {
                "status": {
                    "hp": {
                        "value": self.player.status.hp.value,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "shield": {
                        "value": self.player.status.shield.value,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "is_knocked": {
                        "value": self.player.status.is_knocked,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                },
                "weapon": {
                    "name": {
                        "value": self.player.weapon.name,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "ammo": {
                        "value": self.player.weapon.ammo.value,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                },
                "inventory": {
                    "materials": {
                        "value": self.player.inventory.materials,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                },
            },
            "world": {
                "storm": {
                    "phase": {
                        "value": self.world.storm.phase,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "damage": {
                        "value": self.world.storm.damage,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "in_storm": {
                        "value": self.world.storm.in_storm,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "is_shrinking": {
                        "value": self.world.storm.is_shrinking,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                    "next_circle_distance": {
                        "value": self.world.storm.next_circle_distance,
                        "source": "domain",
                        "confidence": 1.0,
                        "ts_ms": self.updated_at_ms,
                    },
                },
            },
            "session": {
                "phase": {
                    "value": self.session.phase.value,
                    "source": "domain",
                    "confidence": 1.0,
                    "ts_ms": self.updated_at_ms,
                },
                "inactivity_duration_ms": {
                    "value": self.session.inactivity_duration_ms,
                    "source": "domain",
                    "confidence": 1.0,
                    "ts_ms": self.updated_at_ms,
                },
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        """Create from dictionary."""
        player_data = data.get("player", {})
        status_data = player_data.get("status", {})
        weapon_data = player_data.get("weapon", {})
        inventory_data = player_data.get("inventory", {})

        world_data = data.get("world", {})
        storm_data = world_data.get("storm", {})

        session_data = data.get("session", {})

        # Extract values from nested state value objects
        def extract_value(obj: Any, key: str = "value") -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return obj

        hp_value = extract_value(status_data.get("hp", {}), "value")
        shield_value = extract_value(status_data.get("shield", {}), "value")
        is_knocked = extract_value(status_data.get("is_knocked", {}), "value")

        return cls(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=hp_value if hp_value is not None else 100),
                    shield=Shield(value=shield_value if shield_value is not None else 0),
                    is_knocked=is_knocked if is_knocked is not None else False,
                ),
                weapon=WeaponInfo(
                    name=extract_value(weapon_data.get("name", {}), "value"),
                    ammo=Ammo(value=extract_value(weapon_data.get("ammo", {}), "value") or 0),
                ),
                inventory=InventoryInfo(
                    materials=extract_value(inventory_data.get("materials", {}), "value") or 0,
                ),
            ),
            world=WorldInfo(
                storm=StormInfo(
                    phase=extract_value(storm_data.get("phase", {}), "value"),
                    damage=extract_value(storm_data.get("damage", {}), "value"),
                    in_storm=extract_value(storm_data.get("in_storm", {}), "value") or False,
                    is_shrinking=extract_value(storm_data.get("is_shrinking", {}), "value") or False,
                    next_circle_distance=extract_value(storm_data.get("next_circle_distance", {}), "value"),
                ),
            ),
            session=Session.from_dict({
                "phase": extract_value(session_data.get("phase", {}), "value"),
                "inactivity_duration_ms": extract_value(session_data.get("inactivity_duration_ms", {}), "value") or 0,
            }),
            updated_at_ms=get_timestamp_ms(),
        )

    @classmethod
    def empty(cls) -> "GameState":
        """Create empty/default game state."""
        return cls()
