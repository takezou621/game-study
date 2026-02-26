"""JSON output formatter."""

import json
from datetime import datetime
from typing import Any


class JsonFormatter:
    """Format output as JSON."""

    def __init__(self, pretty: bool = True):
        """
        Initialize JSON formatter.

        Args:
            pretty: Whether to pretty-print output
        """
        self.pretty = pretty

    def format(self, data: dict[str, Any]) -> str:
        """
        Format data as JSON.

        Args:
            data: Data to format

        Returns:
            JSON string
        """
        if self.pretty:
            return json.dumps(data, indent=2, default=self._json_default)
        return json.dumps(data, default=self._json_default)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Default JSON serializer."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def format_event(self, event_type: str, data: dict[str, Any]) -> str:
        """
        Format an event for streaming output.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            JSON string
        """
        return self.format({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        })

    def format_state(self, state: dict[str, Any]) -> str:
        """Format game state."""
        return self.format({
            "type": "state",
            "timestamp": datetime.now().isoformat(),
            "state": state,
        })

    def format_trigger(self, trigger_id: str, trigger_name: str, template: str) -> str:
        """Format trigger event."""
        return self.format({
            "type": "trigger",
            "timestamp": datetime.now().isoformat(),
            "trigger_id": trigger_id,
            "trigger_name": trigger_name,
            "template": template,
        })

    def format_response(self, text: str, duration_ms: int | None = None) -> str:
        """Format voice response."""
        return self.format({
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "duration_ms": duration_ms,
        })

    def format_error(self, message: str, details: dict[str, Any] | None = None) -> str:
        """Format error message."""
        return self.format({
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "details": details,
        })
