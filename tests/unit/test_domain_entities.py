"""Tests for domain entities: GameState, PlayerStatus, Session."""

import importlib.util
from pathlib import Path

import pytest

# Direct module imports to avoid package __init__.py dependencies
SRC_PATH = Path(__file__).parent.parent.parent / "src"

# Load domain entities
game_state_spec = importlib.util.spec_from_file_location(
    "game_state",
    SRC_PATH / "domain" / "entities" / "game_state.py",
)
game_state_module = importlib.util.module_from_spec(game_state_spec)
game_state_spec.loader.exec_module(game_state_module)

player_spec = importlib.util.spec_from_file_location(
    "player",
    SRC_PATH / "domain" / "entities" / "player.py",
)
player_module = importlib.util.module_from_spec(player_spec)
player_spec.loader.exec_module(player_module)

session_spec = importlib.util.spec_from_file_location(
    "session",
    SRC_PATH / "domain" / "entities" / "session.py",
)
session_module = importlib.util.module_from_spec(session_spec)
session_spec.loader.exec_module(session_module)

# Load value objects needed by entities
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

# Import classes
GameState = game_state_module.GameState
Player = game_state_module.Player
WeaponInfo = game_state_module.WeaponInfo
InventoryInfo = game_state_module.InventoryInfo
StormInfo = game_state_module.StormInfo
WorldInfo = game_state_module.WorldInfo
MOVEMENT_STATE_COMBAT = game_state_module.MOVEMENT_STATE_COMBAT
MOVEMENT_STATE_NON_COMBAT = game_state_module.MOVEMENT_STATE_NON_COMBAT

PlayerStatus = player_module.PlayerStatus
HP = health_module.HP
Shield = health_module.Shield
Ammo = ammo_module.Ammo

Session = session_module.Session
SessionPhase = session_module.SessionPhase


class TestWeaponInfo:
    """Tests for WeaponInfo class."""

    def test_init_defaults(self):
        """Test WeaponInfo initialization with defaults."""
        weapon = WeaponInfo()
        assert weapon.name is None
        assert weapon.ammo.value == 0

    def test_init_with_values(self):
        """Test WeaponInfo initialization with values."""
        weapon = WeaponInfo(name="Assault Rifle", ammo=Ammo(value=30))
        assert weapon.name == "Assault Rifle"
        assert weapon.ammo.value == 30

    def test_to_dict(self):
        """Test to_dict method."""
        weapon = WeaponInfo(name="Shotgun", ammo=Ammo(value=8))
        result = weapon.to_dict()
        assert result["name"] == "Shotgun"
        assert result["ammo"]["value"] == 8


class TestInventoryInfo:
    """Tests for InventoryInfo class."""

    def test_init_defaults(self):
        """Test InventoryInfo initialization with defaults."""
        inventory = InventoryInfo()
        assert inventory.materials == 0

    def test_init_with_values(self):
        """Test InventoryInfo initialization with values."""
        inventory = InventoryInfo(materials=500)
        assert inventory.materials == 500

    def test_to_dict(self):
        """Test to_dict method."""
        inventory = InventoryInfo(materials=300)
        result = inventory.to_dict()
        assert result["materials"] == 300


class TestStormInfo:
    """Tests for StormInfo class."""

    def test_init_defaults(self):
        """Test StormInfo initialization with defaults."""
        storm = StormInfo()
        assert storm.phase is None
        assert storm.damage is None
        assert storm.in_storm is False
        assert storm.is_shrinking is False
        assert storm.next_circle_distance is None

    def test_init_with_values(self):
        """Test StormInfo initialization with values."""
        storm = StormInfo(
            phase=3,
            damage=2.5,
            in_storm=True,
            is_shrinking=True,
            next_circle_distance=150.0,
        )
        assert storm.phase == 3
        assert storm.damage == 2.5
        assert storm.in_storm is True
        assert storm.is_shrinking is True
        assert storm.next_circle_distance == 150.0

    def test_to_dict(self):
        """Test to_dict method."""
        storm = StormInfo(phase=2, in_storm=True)
        result = storm.to_dict()
        assert result["phase"] == 2
        assert result["in_storm"] is True


class TestWorldInfo:
    """Tests for WorldInfo class."""

    def test_init_defaults(self):
        """Test WorldInfo initialization with defaults."""
        world = WorldInfo()
        assert world.storm.in_storm is False

    def test_to_dict(self):
        """Test to_dict method."""
        world = WorldInfo(storm=StormInfo(phase=3, in_storm=True))
        result = world.to_dict()
        assert result["storm"]["phase"] == 3


class TestPlayerEntity:
    """Tests for Player aggregate class."""

    def test_init_defaults(self):
        """Test Player initialization with defaults."""
        player = Player()
        assert player.status.hp.value == 100
        assert player.status.shield.value == 0
        assert player.weapon.name is None
        assert player.inventory.materials == 0

    def test_to_dict(self):
        """Test Player to_dict method."""
        player = Player(
            status=PlayerStatus(hp=HP(value=75), shield=Shield(value=50)),
            weapon=WeaponInfo(name="AR", ammo=Ammo(value=30)),
            inventory=InventoryInfo(materials=200),
        )
        result = player.to_dict()
        assert result["status"]["hp"]["value"] == 75
        assert result["status"]["shield"]["value"] == 50
        assert result["weapon"]["name"] == "AR"
        assert result["inventory"]["materials"] == 200


class TestGameState:
    """Tests for GameState aggregate root."""

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a test game state."""
        return GameState(
            player=Player(
                status=PlayerStatus(
                    hp=HP(value=100),
                    shield=Shield(value=50),
                    is_knocked=False,
                ),
            ),
        )

    def test_init_defaults(self):
        """Test GameState initialization with defaults."""
        state = GameState()
        assert state.player.status.hp.value == 100
        assert state.player.status.shield.value == 0
        assert state.world.storm.in_storm is False
        assert state.session.phase == SessionPhase.LOBBY

    def test_is_combat_when_hp_low(self, game_state: GameState):
        """Test is_combat returns True when HP is low."""
        # HP >= 50 means not combat
        assert game_state.is_combat is False

        # Create state with low HP
        low_hp_state = GameState(
            player=Player(
                status=PlayerStatus(hp=HP(value=30)),
            ),
        )
        assert low_hp_state.is_combat is True

    def test_is_combat_when_in_storm(self):
        """Test is_combat returns True when in storm."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True)),
        )
        assert state.is_combat is True

    def test_movement_state(self, game_state: GameState):
        """Test movement_state property."""
        assert game_state.movement_state == MOVEMENT_STATE_NON_COMBAT

        combat_state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=30))),
        )
        assert combat_state.movement_state == MOVEMENT_STATE_COMBAT

    def test_needs_attention_when_low_hp(self):
        """Test needs_attention returns True when HP is low."""
        state = GameState(
            player=Player(status=PlayerStatus(hp=HP(value=20))),
        )
        assert state.needs_attention is True

    def test_needs_attention_when_in_storm(self):
        """Test needs_attention returns True when in storm."""
        state = GameState(
            world=WorldInfo(storm=StormInfo(in_storm=True)),
        )
        assert state.needs_attention is True

    def test_needs_attention_when_knocked(self):
        """Test needs_attention returns True when knocked."""
        state = GameState(
            player=Player(status=PlayerStatus(is_knocked=True)),
        )
        assert state.needs_attention is True

    def test_needs_attention_when_healthy(self, game_state: GameState):
        """Test needs_attention returns False when healthy."""
        assert game_state.needs_attention is False

    def test_with_hp(self, game_state: GameState):
        """Test with_hp creates new state with updated HP."""
        new_state = game_state.with_hp(50)
        assert new_state.player.status.hp.value == 50
        # Original should be unchanged
        assert game_state.player.status.hp.value == 100

    def test_with_shield(self, game_state: GameState):
        """Test with_shield creates new state with updated shield."""
        new_state = game_state.with_shield(100)
        assert new_state.player.status.shield.value == 100
        assert game_state.player.status.shield.value == 50

    def test_with_knocked(self, game_state: GameState):
        """Test with_knocked creates new state with knocked status."""
        new_state = game_state.with_knocked(True)
        assert new_state.player.status.is_knocked is True
        assert game_state.player.status.is_knocked is False

    def test_with_weapon(self, game_state: GameState):
        """Test with_weapon creates new state with weapon info."""
        new_state = game_state.with_weapon("SMG", 25)
        assert new_state.player.weapon.name == "SMG"
        assert new_state.player.weapon.ammo.value == 25

    def test_with_materials(self, game_state: GameState):
        """Test with_materials creates new state with materials."""
        new_state = game_state.with_materials(500)
        assert new_state.player.inventory.materials == 500

    def test_with_storm(self, game_state: GameState):
        """Test with_storm creates new state with storm info."""
        new_state = game_state.with_storm(
            phase=3,
            damage=2.5,
            in_storm=True,
            is_shrinking=True,
        )
        assert new_state.world.storm.phase == 3
        assert new_state.world.storm.damage == 2.5
        assert new_state.world.storm.in_storm is True
        assert new_state.world.storm.is_shrinking is True

    def test_to_dict(self, game_state: GameState):
        """Test to_dict returns correct structure."""
        result = game_state.to_dict()
        assert "player" in result
        assert "status" in result["player"]
        assert "hp" in result["player"]["status"]
        assert result["player"]["status"]["hp"]["value"] == 100

    def test_from_dict(self):
        """Test from_dict creates correct state."""
        data = {
            "player": {
                "status": {
                    "hp": {"value": 75},
                    "shield": {"value": 25},
                    "is_knocked": False,
                },
                "weapon": {
                    "name": {"value": "Rifle"},
                    "ammo": {"value": 20},
                },
                "inventory": {
                    "materials": {"value": 300},
                },
            },
            "world": {
                "storm": {
                    "phase": {"value": 2},
                    "in_storm": {"value": True},
                },
            },
            "session": {
                "phase": {"value": "mid_game"},
            },
        }
        state = GameState.from_dict(data)
        assert state.player.status.hp.value == 75
        assert state.player.status.shield.value == 25
        assert state.player.weapon.name == "Rifle"
        assert state.player.weapon.ammo.value == 20
        assert state.world.storm.phase == 2
        assert state.world.storm.in_storm is True
        assert state.session.phase == SessionPhase.MID_GAME

    def test_from_dict_with_raw_values(self):
        """Test from_dict handles raw values (not nested)."""
        data = {
            "player": {
                "status": {
                    "hp": 80,
                    "shield": 40,
                    "is_knocked": True,
                },
            },
        }
        state = GameState.from_dict(data)
        assert state.player.status.hp.value == 80
        assert state.player.status.shield.value == 40
        assert state.player.status.is_knocked is True

    def test_empty(self):
        """Test empty factory method."""
        state = GameState.empty()
        assert state.player.status.hp.value == 100
        assert state.player.status.shield.value == 0
        assert state.session.phase == SessionPhase.LOBBY


class TestPlayerStatus:
    """Tests for PlayerStatus entity."""

    @pytest.fixture
    def player_status(self) -> PlayerStatus:
        """Create a test player status."""
        return PlayerStatus(
            hp=HP(value=100),
            shield=Shield(value=50),
            is_knocked=False,
        )

    def test_init_defaults(self):
        """Test PlayerStatus initialization with defaults."""
        status = PlayerStatus()
        assert status.hp.value == 100
        assert status.shield.value == 0
        assert status.is_knocked is False

    def test_is_alive(self, player_status: PlayerStatus):
        """Test is_alive property."""
        assert player_status.is_alive is True

        # HP = 0 means dead
        dead_status = PlayerStatus(hp=HP(value=0))
        assert dead_status.is_alive is False

        # Knocked means not alive
        knocked_status = PlayerStatus(is_knocked=True)
        assert knocked_status.is_alive is False

    def test_total_protection(self, player_status: PlayerStatus):
        """Test total_protection property."""
        assert player_status.total_protection == 150

    def test_effective_health(self, player_status: PlayerStatus):
        """Test effective_health property."""
        assert player_status.effective_health == 150

    def test_take_damage_shield_only(self, player_status: PlayerStatus):
        """Test take_damage when shield absorbs all."""
        new_status = player_status.take_damage(30)
        assert new_status.shield.value == 20
        assert new_status.hp.value == 100

    def test_take_damage_shield_and_hp(self, player_status: PlayerStatus):
        """Test take_damage when damage exceeds shield."""
        new_status = player_status.take_damage(75)
        assert new_status.shield.value == 0
        assert new_status.hp.value == 75

    def test_take_damage_hp_only(self):
        """Test take_damage when no shield."""
        status = PlayerStatus(hp=HP(value=100), shield=Shield(value=0))
        new_status = status.take_damage(30)
        assert new_status.hp.value == 70
        assert new_status.shield.value == 0

    def test_take_damage_lethal(self, player_status: PlayerStatus):
        """Test take_damage cannot go below 0 HP."""
        new_status = player_status.take_damage(200)
        assert new_status.hp.value == 0

    def test_heal(self, player_status: PlayerStatus):
        """Test heal method."""
        damaged = PlayerStatus(hp=HP(value=50))
        healed = damaged.heal(30)
        assert healed.hp.value == 80

    def test_heal_cannot_exceed_max(self):
        """Test heal cannot exceed max HP."""
        status = PlayerStatus(hp=HP(value=90))
        healed = status.heal(30)
        assert healed.hp.value == 100

    def test_add_shield(self, player_status: PlayerStatus):
        """Test add_shield method."""
        new_status = player_status.add_shield(25)
        assert new_status.shield.value == 75

    def test_add_shield_cannot_exceed_max(self, player_status: PlayerStatus):
        """Test add_shield cannot exceed max."""
        new_status = player_status.add_shield(100)
        assert new_status.shield.value == 100

    def test_knock(self, player_status: PlayerStatus):
        """Test knock method."""
        knocked = player_status.knock()
        assert knocked.is_knocked is True
        assert player_status.is_knocked is False  # Original unchanged

    def test_revive(self):
        """Test revive method."""
        knocked = PlayerStatus(is_knocked=True)
        revived = knocked.revive()
        assert revived.is_knocked is False
        assert revived.hp.value == 100

    def test_revive_with_custom_hp(self):
        """Test revive with custom HP restore."""
        knocked = PlayerStatus(is_knocked=True)
        revived = knocked.revive(hp_restore=50)
        assert revived.hp.value == 50

    def test_to_dict(self, player_status: PlayerStatus):
        """Test to_dict method."""
        result = player_status.to_dict()
        assert result["hp"]["value"] == 100
        assert result["shield"]["value"] == 50
        assert result["is_knocked"] is False

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "hp": {"value": 75, "max_value": 100},
            "shield": {"value": 25, "max_value": 100},
            "is_knocked": True,
        }
        status = PlayerStatus.from_dict(data)
        assert status.hp.value == 75
        assert status.shield.value == 25
        assert status.is_knocked is True


class TestSession:
    """Tests for Session entity."""

    @pytest.fixture
    def session(self) -> Session:
        """Create a test session."""
        return Session(
            phase=SessionPhase.MID_GAME,
            inactivity_duration_ms=0,
            started_at_ms=1000000,
            last_activity_ms=1000000,
        )

    def test_init_defaults(self):
        """Test Session initialization with defaults."""
        sess = Session()
        assert sess.phase == SessionPhase.LOBBY
        assert sess.inactivity_duration_ms == 0

    def test_session_phase_enum(self):
        """Test SessionPhase enum values."""
        assert SessionPhase.LOBBY.value == "lobby"
        assert SessionPhase.BUS.value == "bus"
        assert SessionPhase.EARLY_GAME.value == "early_game"
        assert SessionPhase.MID_GAME.value == "mid_game"
        assert SessionPhase.LATE_GAME.value == "late_game"
        assert SessionPhase.END_GAME.value == "end_game"

    def test_is_inactive(self, session: Session):
        """Test is_inactive property."""
        assert session.is_inactive is False

        inactive_session = Session(inactivity_duration_ms=35000)
        assert inactive_session.is_inactive is True

    def test_record_activity(self, session: Session):
        """Test record_activity method."""
        new_session = session.record_activity()
        # Inactivity should be reset to 0
        assert new_session.inactivity_duration_ms == 0
        # Last activity should be updated (greater than original)
        assert new_session.last_activity_ms > session.last_activity_ms
        # Original unchanged
        assert session.last_activity_ms == 1000000

    def test_update_inactivity(self, session: Session):
        """Test update_inactivity method."""
        new_session = session.update_inactivity()
        # Inactivity should be updated (greater than 0 since time has passed)
        assert new_session.inactivity_duration_ms >= 0
        # Should be calculated from current time - last_activity
        # Since last_activity was 1000000, inactivity should be approximately
        # current_time - 1000000
        assert new_session.inactivity_duration_ms > 0

    def test_set_phase(self, session: Session):
        """Test set_phase method."""
        new_session = session.set_phase(SessionPhase.LATE_GAME)
        assert new_session.phase == SessionPhase.LATE_GAME
        assert session.phase == SessionPhase.MID_GAME

    def test_to_dict(self, session: Session):
        """Test to_dict method."""
        result = session.to_dict()
        assert result["phase"] == "mid_game"
        assert result["inactivity_duration_ms"] == 0
        assert "started_at_ms" in result
        assert "last_activity_ms" in result

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "phase": "late_game",
            "inactivity_duration_ms": 10000,
            "started_at_ms": 1000000,
            "last_activity_ms": 1050000,
        }
        session = Session.from_dict(data)
        assert session.phase == SessionPhase.LATE_GAME
        assert session.inactivity_duration_ms == 10000

    def test_from_dict_invalid_phase(self):
        """Test from_dict handles invalid phase."""
        data = {"phase": "invalid_phase"}
        session = Session.from_dict(data)
        assert session.phase == SessionPhase.LOBBY

    def test_from_dict_defaults(self):
        """Test from_dict with empty data uses defaults."""
        session = Session.from_dict({})
        assert session.phase == SessionPhase.LOBBY
        assert session.inactivity_duration_ms == 0
