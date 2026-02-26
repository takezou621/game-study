"""Game analyzer service for analyzing game state and providing insights."""

from dataclasses import dataclass, field
from typing import Any

from domain.entities.game_state import GameState
from domain.services.state_validator import StateValidator


@dataclass
class GameAnalysis:
    """Analysis result for game state."""

    urgency_level: int  # 0-3
    is_combat: bool
    needs_attention: bool
    recommendations: list[str] = field(default_factory=list)
    key_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameAnalyzerService:
    """Service for analyzing game state and providing insights."""

    state_validator: StateValidator = field(default_factory=StateValidator)

    def analyze(self, state: GameState) -> GameAnalysis:
        """Analyze game state and provide insights.

        Args:
            state: Current game state

        Returns:
            GameAnalysis with urgency, recommendations, etc.
        """
        urgency = self.state_validator.calculate_urgency(state)
        is_combat = state.is_combat
        needs_attention = state.needs_attention

        recommendations = self._generate_recommendations(state)
        key_metrics = self._extract_key_metrics(state)

        return GameAnalysis(
            urgency_level=urgency,
            is_combat=is_combat,
            needs_attention=needs_attention,
            recommendations=recommendations,
            key_metrics=key_metrics,
        )

    def _generate_recommendations(self, state: GameState) -> list[str]:
        """Generate recommendations based on state.

        Args:
            state: Current game state

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # HP-based recommendations
        if state.player.status.hp.is_critical:
            recommendations.append("Heal immediately - critical HP!")
        elif state.player.status.hp.is_low:
            recommendations.append("Consider healing soon")

        # Storm recommendations
        if state.world.storm.in_storm:
            if state.world.storm.damage and state.world.storm.damage > 3:
                recommendations.append("Get out of storm NOW - heavy damage!")
            else:
                recommendations.append("Move towards safe zone")

        if state.world.storm.is_shrinking:
            recommendations.append("Storm is shrinking - check circle position")

        # Knocked state
        if state.player.status.is_knocked:
            recommendations.append("Call for revive or crawl to teammate")

        # Ammo check
        if state.player.weapon.ammo.is_low and state.player.weapon.ammo.value > 0:
            recommendations.append("Low ammo - reload when safe")
        elif state.player.weapon.ammo.is_empty:
            recommendations.append("Out of ammo - switch weapons!")

        # Materials
        if state.player.inventory.materials < 100:
            recommendations.append("Farm materials when possible")

        return recommendations

    def _extract_key_metrics(self, state: GameState) -> dict[str, Any]:
        """Extract key metrics from state.

        Args:
            state: Current game state

        Returns:
            Dictionary of key metrics
        """
        return {
            "hp": state.player.status.hp.value,
            "shield": state.player.status.shield.value,
            "total_health": state.player.status.effective_health,
            "is_knocked": state.player.status.is_knocked,
            "in_storm": state.world.storm.in_storm,
            "storm_phase": state.world.storm.phase,
            "weapon": state.player.weapon.name,
            "ammo": state.player.weapon.ammo.value,
            "materials": state.player.inventory.materials,
            "session_phase": state.session.phase.value,
        }

    def get_movement_state(self, state: GameState) -> str:
        """Get movement state for template selection.

        Args:
            state: Current game state

        Returns:
            Movement state string ("combat" or "non_combat")
        """
        return state.movement_state

    def compare_states(
        self,
        previous: GameState,
        current: GameState,
    ) -> dict[str, Any]:
        """Compare two game states and identify changes.

        Args:
            previous: Previous state
            current: Current state

        Returns:
            Dictionary of changes
        """
        changes = {}

        # HP change
        if previous.player.status.hp.value != current.player.status.hp.value:
            hp_diff = current.player.status.hp.value - previous.player.status.hp.value
            changes["hp"] = {
                "old": previous.player.status.hp.value,
                "new": current.player.status.hp.value,
                "diff": hp_diff,
                "is_damage": hp_diff < 0,
            }

        # Shield change
        if previous.player.status.shield.value != current.player.status.shield.value:
            shield_diff = current.player.status.shield.value - previous.player.status.shield.value
            changes["shield"] = {
                "old": previous.player.status.shield.value,
                "new": current.player.status.shield.value,
                "diff": shield_diff,
            }

        # Knocked state change
        if previous.player.status.is_knocked != current.player.status.is_knocked:
            changes["knocked"] = {
                "old": previous.player.status.is_knocked,
                "new": current.player.status.is_knocked,
            }

        # Storm state change
        if previous.world.storm.in_storm != current.world.storm.in_storm:
            changes["in_storm"] = {
                "old": previous.world.storm.in_storm,
                "new": current.world.storm.in_storm,
            }

        return changes

    def should_suppress_response(
        self,
        state: GameState,
        last_response_ms: int,
        min_interval_ms: int = 3000,
    ) -> bool:
        """Determine if responses should be suppressed.

        Args:
            state: Current game state
            last_response_ms: Time of last response
            min_interval_ms: Minimum interval between responses

        Returns:
            True if should suppress
        """
        from utils.time import get_timestamp_ms

        elapsed = get_timestamp_ms() - last_response_ms

        # Always allow urgent responses
        if state.needs_attention and elapsed >= 1000:
            return False

        return elapsed < min_interval_ms
