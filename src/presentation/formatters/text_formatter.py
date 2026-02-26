"""Text output formatter."""

from datetime import datetime
from typing import Any


class TextFormatter:
    """Format output as human-readable text."""

    def __init__(self, use_colors: bool = True):
        """
        Initialize text formatter.

        Args:
            use_colors: Whether to use ANSI colors
        """
        self.use_colors = use_colors

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color."""
        if not self.use_colors:
            return text

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def format_state(self, state: dict[str, Any]) -> str:
        """Format game state for display."""
        lines = []
        lines.append(self._color("=== Game State ===", "cyan"))
        lines.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")

        # Player status
        player = state.get("player", {})
        status = player.get("status", {})

        hp_val = status.get("hp", {}).get("value")
        shield_val = status.get("shield", {}).get("value")

        hp_str = self._color(f"HP: {hp_val}", "green" if hp_val and hp_val > 50 else "red")
        shield_str = self._color(f"Shield: {shield_val}", "blue")

        lines.append(f"{hp_str} | {shield_str}")

        # Weapon
        weapon = player.get("weapon", {})
        weapon_name = weapon.get("name", {}).get("value", "None")
        ammo = weapon.get("ammo", {}).get("value", 0)
        lines.append(f"Weapon: {weapon_name} | Ammo: {ammo}")

        # World
        world = state.get("world", {})
        storm = world.get("storm", {})
        in_storm = storm.get("in_storm", {}).get("value", False)
        if in_storm:
            lines.append(self._color("⚠ IN STORM!", "red"))

        return "\n".join(lines)

    def format_trigger(self, trigger_id: str, trigger_name: str, template: str) -> str:
        """Format trigger event."""
        lines = []
        lines.append(self._color(f"[TRIGGER] {trigger_name}", "yellow"))
        lines.append(f"  ID: {trigger_id}")
        lines.append(f"  Response: {template}")
        return "\n".join(lines)

    def format_response(self, text: str, duration_ms: int | None = None) -> str:
        """Format voice response."""
        duration_str = f" ({duration_ms}ms)" if duration_ms else ""
        return self._color(f">> {text}{duration_str}", "green")

    def format_error(self, message: str, details: dict[str, Any] | None = None) -> str:
        """Format error message."""
        lines = []
        lines.append(self._color(f"ERROR: {message}", "red"))
        if details:
            for key, value in details.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def format_summary(self, stats: dict[str, Any]) -> str:
        """Format session summary."""
        lines = []
        lines.append(self._color("=== Session Summary ===", "cyan"))
        lines.append(f"Duration: {stats.get('duration_seconds', 0):.1f}s")
        lines.append(f"Frames processed: {stats.get('frames_processed', 0)}")
        lines.append(f"Triggers fired: {stats.get('triggers_fired', 0)}")
        lines.append(f"Responses generated: {stats.get('responses_generated', 0)}")
        return "\n".join(lines)

    def format_startup(self, config: dict[str, Any]) -> str:
        """Format startup message."""
        lines = []
        lines.append(self._color("=" * 50, "cyan"))
        lines.append(self._color("  AI Game Coach", "bold"))
        lines.append(self._color("=" * 50, "cyan"))
        lines.append(f"Input: {config.get('input_type', 'unknown')}")
        lines.append(f"Audio: {'enabled' if config.get('audio_enabled') else 'disabled'}")
        lines.append(f"Voice: {config.get('voice', 'default')}")
        lines.append("")
        lines.append(self._color("Press Ctrl+C to stop", "yellow"))
        return "\n".join(lines)
