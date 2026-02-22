"""Game State builder from vision detections."""

from typing import Any

from utils.time import get_timestamp_ms

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create a minimal BaseModel fallback if pydantic is not available
    class BaseModel:
        """Fallback BaseModel for when pydantic is not available."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return self.__dict__

        @classmethod
        def model_validate(cls, data):
            return cls(**data)

    def Field(default=None, **kwargs):
        return default


# Movement state constants
MOVEMENT_STATE_COMBAT = "combat"
MOVEMENT_STATE_NON_COMBAT = "non_combat"
MOVEMENT_STATES = [MOVEMENT_STATE_COMBAT, MOVEMENT_STATE_NON_COMBAT]


if PYDANTIC_AVAILABLE:
    class StateValueModel(BaseModel):
        """Pydantic model for state value with metadata."""

        value: Any = Field(..., description="The actual value")
        source: str = Field(..., description="Source of the value (roi, yolo, ocr, etc.)")
        confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
        ts_ms: int = Field(..., description="Timestamp in milliseconds")

        @field_validator('confidence')
        @classmethod
        def validate_confidence(cls, v):
            """Validate confidence is in valid range."""
            if not 0.0 <= v <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
            return v

        class Config:
            arbitrary_types_allowed = True


    class PlayerStatusModel(BaseModel):
        """Pydantic model for player status."""

        hp: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=100, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))
        shield: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=0, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))
        is_knocked: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=False, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))


    class WeaponInfoModel(BaseModel):
        """Pydantic model for weapon information."""

        name: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))
        ammo: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))


    class InventoryInfoModel(BaseModel):
        """Pydantic model for inventory information."""

        materials: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))


    class StormInfoModel(BaseModel):
        """Pydantic model for storm information."""

        phase: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))
        damage: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))
        in_storm: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=False, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))
        is_shrinking: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=False, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))
        next_circle_distance: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))


    class SessionInfoModel(BaseModel):
        """Pydantic model for session information."""

        phase: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))
        inactivity_duration_ms: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=0, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))


    class PlayerModel(BaseModel):
        """Pydantic model for player state."""

        status: PlayerStatusModel = Field(default_factory=PlayerStatusModel)
        weapon: WeaponInfoModel = Field(default_factory=WeaponInfoModel)
        inventory: InventoryInfoModel = Field(default_factory=InventoryInfoModel)


    class WorldModel(BaseModel):
        """Pydantic model for world state."""

        storm: StormInfoModel = Field(default_factory=StormInfoModel)


    class SessionModel(BaseModel):
        """Pydantic model for session state."""

        phase: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=None, source="default", confidence=0.0, ts_ms=get_timestamp_ms()
        ))
        inactivity_duration_ms: StateValueModel = Field(default_factory=lambda: StateValueModel(
            value=0, source="default", confidence=1.0, ts_ms=get_timestamp_ms()
        ))


    class GameStateModel(BaseModel):
        """Pydantic model for complete game state."""

        player: PlayerModel = Field(default_factory=PlayerModel)
        world: WorldModel = Field(default_factory=WorldModel)
        session: SessionModel = Field(default_factory=SessionModel)


class StateBuilder:
    """Build Game State JSON from vision detections."""

    def __init__(self):
        """Initialize state builder."""
        self.current_state = self._create_empty_state()

    def _create_empty_state(self) -> dict[str, Any]:
        """
        Create empty game state template.

        Returns:
            Empty state dictionary
        """
        return {
            "player": {
                "status": {
                    "hp": {"value": 100, "source": "default", "confidence": 1.0, "ts_ms": get_timestamp_ms()},
                    "shield": {"value": 0, "source": "default", "confidence": 1.0, "ts_ms": get_timestamp_ms()},
                    "is_knocked": {"value": False, "source": "default", "confidence": 1.0, "ts_ms": get_timestamp_ms()},
                },
                "weapon": {
                    "name": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                    "ammo": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                },
                "inventory": {
                    "materials": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                },
            },
            "world": {
                "storm": {
                    "phase": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                    "damage": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                    "in_storm": {"value": False, "source": "default", "confidence": 1.0, "ts_ms": get_timestamp_ms()},
                    "is_shrinking": {"value": False, "source": "default", "confidence": 1.0, "ts_ms": get_timestamp_ms()},
                    "next_circle_distance": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                },
            },
            "session": {
                "phase": {"value": None, "source": "default", "confidence": 0.0, "ts_ms": get_timestamp_ms()},
                "inactivity_duration_ms": {"value": 0, "source": "default", "confidence": 1.0, "ts_ms": get_timestamp_ms()},
            },
        }

    def _validate_state_value(
        self,
        value: Any,
        source: str,
        confidence: float
    ) -> dict[str, Any]:
        """
        Validate a state value using Pydantic if available.

        Args:
            value: The value to validate
            source: Source of the value
            confidence: Confidence score (0.0-1.0)

        Returns:
            Validated state value dictionary

        Raises:
            ValueError: If validation fails (when using Pydantic)
        """
        # Validate confidence is in range
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        if PYDANTIC_AVAILABLE:
            try:
                validated = StateValueModel(
                    value=value,
                    source=source,
                    confidence=confidence,
                    ts_ms=get_timestamp_ms()
                )
                return validated.model_dump()
            except Exception:
                # Fallback to dict if validation fails for optional fields
                return {
                    "value": value,
                    "source": source,
                    "confidence": confidence,
                    "ts_ms": get_timestamp_ms()
                }
        else:
            return {
                "value": value,
                "source": source,
                "confidence": confidence,
                "ts_ms": get_timestamp_ms()
            }

    def update_field(
        self,
        path: str,
        value: Any,
        source: str,
        confidence: float
    ) -> None:
        """
        Update a specific field in the state.

        Args:
            path: Dot-separated path (e.g., "player.status.hp")
            value: New value
            source: Source of the value (roi, yolo, ocr, etc.)
            confidence: Confidence score (0.0-1.0)

        Raises:
            ValueError: If confidence is out of valid range
        """
        keys = path.split('.')
        current = self.current_state

        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]

        last_key = keys[-1]
        if last_key in current:
            # Update the field with validation
            validated_value = self._validate_state_value(value, source, confidence)
            current[last_key] = validated_value

    def update_hp(self, value: int, source: str, confidence: float) -> None:
        """
        Update HP value.

        Args:
            value: HP value (typically 0-100)
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("player.status.hp", value, source, confidence)

    def update_shield(self, value: int, source: str, confidence: float) -> None:
        """
        Update Shield value.

        Args:
            value: Shield value (typically 0-100)
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("player.status.shield", value, source, confidence)

    def update_knocked(self, value: bool, source: str, confidence: float) -> None:
        """
        Update knocked status.

        Args:
            value: Whether player is knocked
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("player.status.is_knocked", value, source, confidence)

    def update_weapon_name(self, value: str, source: str, confidence: float) -> None:
        """
        Update weapon name.

        Args:
            value: Weapon name
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("player.weapon.name", value, source, confidence)

    def update_ammo(self, value: int, source: str, confidence: float) -> None:
        """
        Update ammo count.

        Args:
            value: Ammo count
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("player.weapon.ammo", value, source, confidence)

    def update_materials(self, value: int, source: str, confidence: float) -> None:
        """
        Update materials count.

        Args:
            value: Materials count
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("player.inventory.materials", value, source, confidence)

    def update_storm_phase(self, value: int, source: str, confidence: float) -> None:
        """
        Update storm phase.

        Args:
            value: Storm phase number
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("world.storm.phase", value, source, confidence)

    def update_storm_damage(self, value: float, source: str, confidence: float) -> None:
        """
        Update storm damage.

        Args:
            value: Storm damage per second
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("world.storm.damage", value, source, confidence)

    def update_in_storm(self, value: bool, source: str, confidence: float) -> None:
        """
        Update in-storm status.

        Args:
            value: Whether player is in storm
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("world.storm.in_storm", value, source, confidence)

    def update_storm_shrinking(self, value: bool, source: str, confidence: float) -> None:
        """
        Update storm shrinking status.

        Args:
            value: Whether storm is shrinking
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("world.storm.is_shrinking", value, source, confidence)

    def update_session_phase(self, value: str, source: str, confidence: float) -> None:
        """
        Update session phase.

        Args:
            value: Session phase identifier
            source: Source of the detection
            confidence: Confidence score (0.0-1.0)
        """
        self.update_field("session.phase", value, source, confidence)

    def get_state(self) -> dict[str, Any]:
        """
        Get current state.

        Returns:
            Current game state
        """
        return self.current_state

    def reset(self) -> None:
        """Reset state to empty template."""
        self.current_state = self._create_empty_state()

    def get_movement_state(self) -> str:
        """
        Determine movement state based on current state.

        Returns:
            Movement state: "combat" or "non_combat"
        """
        # Heuristic: if HP < 50 or in storm, consider combat
        hp = self.current_state["player"]["status"]["hp"]["value"]
        in_storm = self.current_state["world"]["storm"]["in_storm"]["value"]

        if (hp is not None and hp < 50) or in_storm:
            return MOVEMENT_STATE_COMBAT
        return MOVEMENT_STATE_NON_COMBAT
