"""Player status entity."""

from dataclasses import dataclass, field
from typing import Any

from domain.value_objects.health import HP, Shield


@dataclass
class PlayerStatus:
    """Player status entity.

    Represents the current status of a player including health,
    shield, and knocked state.
    """

    hp: HP = field(default_factory=lambda: HP(value=100))
    shield: Shield = field(default_factory=lambda: Shield(value=0))
    is_knocked: bool = False

    @property
    def is_alive(self) -> bool:
        """Check if player is alive."""
        return self.hp.value > 0 and not self.is_knocked

    @property
    def total_protection(self) -> int:
        """Get total HP + Shield."""
        return self.hp.value + self.shield.value

    @property
    def effective_health(self) -> int:
        """Get effective health (HP + Shield)."""
        return self.hp.value + self.shield.value

    def take_damage(self, damage: int) -> "PlayerStatus":
        """Apply damage to player (shield first, then HP).

        Args:
            damage: Amount of damage to apply

        Returns:
            New PlayerStatus with updated values
        """
        remaining_damage = damage
        new_shield = self.shield.value
        new_hp = self.hp.value

        # Damage shield first
        if new_shield > 0:
            if remaining_damage >= new_shield:
                remaining_damage -= new_shield
                new_shield = 0
            else:
                new_shield -= remaining_damage
                remaining_damage = 0

        # Remaining damage goes to HP
        if remaining_damage > 0:
            new_hp = max(0, new_hp - remaining_damage)

        return PlayerStatus(
            hp=HP(value=new_hp),
            shield=Shield(value=new_shield),
            is_knocked=self.is_knocked,
        )

    def heal(self, amount: int) -> "PlayerStatus":
        """Heal HP.

        Args:
            amount: Amount to heal

        Returns:
            New PlayerStatus with updated HP
        """
        new_hp = min(self.hp.max_value, self.hp.value + amount)
        return PlayerStatus(
            hp=HP(value=new_hp),
            shield=self.shield,
            is_knocked=self.is_knocked,
        )

    def add_shield(self, amount: int) -> "PlayerStatus":
        """Add shield.

        Args:
            amount: Amount of shield to add

        Returns:
            New PlayerStatus with updated shield
        """
        new_shield = min(self.shield.max_value, self.shield.value + amount)
        return PlayerStatus(
            hp=self.hp,
            shield=Shield(value=new_shield),
            is_knocked=self.is_knocked,
        )

    def knock(self) -> "PlayerStatus":
        """Knock the player.

        Returns:
            New PlayerStatus with knocked state
        """
        return PlayerStatus(
            hp=self.hp,
            shield=self.shield,
            is_knocked=True,
        )

    def revive(self, hp_restore: int = 100) -> "PlayerStatus":
        """Revive the player.

        Args:
            hp_restore: HP to restore on revive

        Returns:
            New PlayerStatus with revived state
        """
        return PlayerStatus(
            hp=HP(value=min(self.hp.max_value, hp_restore)),
            shield=self.shield,
            is_knocked=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hp": self.hp.to_dict(),
            "shield": self.shield.to_dict(),
            "is_knocked": self.is_knocked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerStatus":
        """Create from dictionary."""
        hp_data = data.get("hp", {})
        shield_data = data.get("shield", {})

        return cls(
            hp=HP(
                value=hp_data.get("value", 100),
                max_value=hp_data.get("max_value", 100),
            ),
            shield=Shield(
                value=shield_data.get("value", 0),
                max_value=shield_data.get("max_value", 100),
            ),
            is_knocked=data.get("is_knocked", False),
        )
