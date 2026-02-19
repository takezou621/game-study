"""Session logger with JSONL support."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SessionLogger:
    """Logger for game sessions with JSONL output."""

    def __init__(self, session_dir: str):
        """
        Initialize session logger.

        Args:
            session_dir: Directory for session logs
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.state_log_path = self.session_dir / "state.jsonl"
        self.trigger_log_path = self.session_dir / "triggers.jsonl"
        self.response_log_path = self.session_dir / "responses.jsonl"
        self.error_log_path = self.session_dir / "errors.log"

    def log_state(self, state: Dict[str, Any]) -> None:
        """
        Log game state to JSONL file.

        Args:
            state: Game state dictionary
        """
        with open(self.state_log_path, 'a') as f:
            f.write(json.dumps(state, ensure_ascii=False) + '\n')

    def log_trigger(self, trigger: Dict[str, Any]) -> None:
        """
        Log trigger event to JSONL file.

        Args:
            trigger: Trigger event dictionary
        """
        with open(self.trigger_log_path, 'a') as f:
            f.write(json.dumps(trigger, ensure_ascii=False) + '\n')

    def log_response(self, response: Dict[str, Any]) -> None:
        """
        Log AI response to JSONL file.

        Args:
            response: Response dictionary
        """
        with open(self.response_log_path, 'a') as f:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')

    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """
        Log error to error file.

        Args:
            message: Error message
            exception: Optional exception object
        """
        timestamp = datetime.now().isoformat()
        error_msg = f"[{timestamp}] {message}"
        if exception:
            error_msg += f"\n  Exception: {type(exception).__name__}: {exception}"

        with open(self.error_log_path, 'a') as f:
            f.write(error_msg + '\n')

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get session information.

        Returns:
            Session info dictionary
        """
        return {
            "session_dir": str(self.session_dir),
            "state_log_path": str(self.state_log_path),
            "trigger_log_path": str(self.trigger_log_path),
            "response_log_path": str(self.response_log_path),
            "created_at": datetime.now().isoformat(),
        }
