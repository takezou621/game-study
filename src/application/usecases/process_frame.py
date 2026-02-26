"""Use case for processing video frames."""

from dataclasses import dataclass
from typing import Any, Protocol

from application.dto.frame_dto import DetectionResultDTO, FrameDTO, FrameResultDTO
from application.ports.capture_port import CapturePort


class VisionDetector(Protocol):
    """Protocol for vision detection services."""

    def detect(self, frame: Any) -> list[DetectionResultDTO]:
        """Run detection on frame."""
        ...


class StateBuilder(Protocol):
    """Protocol for state building."""

    def update_from_detections(self, detections: list[DetectionResultDTO]) -> None:
        """Update state from detections."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Get current state."""
        ...


@dataclass
class ProcessFrameUseCase:
    """Use case for processing a single frame through the vision pipeline."""

    capture: CapturePort
    state_builder: StateBuilder
    detectors: list[VisionDetector]
    frame_number: int = 0

    def execute(self, frame: FrameDTO | None = None) -> FrameResultDTO | None:
        """Process a single frame.

        Args:
            frame: Optional frame DTO. If None, reads from capture.

        Returns:
            FrameResultDTO with detections and state changes, or None if no frame
        """
        import time

        start_time = time.time()

        # Get frame if not provided
        if frame is None:
            frame_data = self.capture.read()
            if frame_data is None:
                return None
            frame = FrameDTO(
                frame=frame_data,
                frame_number=self.frame_number,
            )

        self.frame_number = frame.frame_number

        # Run all detectors
        all_detections: list[DetectionResultDTO] = []
        for detector in self.detectors:
            try:
                detections = detector.detect(frame.frame)
                all_detections.extend(detections)
            except Exception:
                # Continue with other detectors on error
                pass

        # Update state from detections
        previous_state = self.state_builder.get_state().copy()
        self.state_builder.update_from_detections(all_detections)
        current_state = self.state_builder.get_state()

        # Calculate state changes
        state_changes = self._calculate_state_changes(previous_state, current_state)

        # Build result
        processing_time_ms = int((time.time() - start_time) * 1000)

        return FrameResultDTO(
            frame_number=frame.frame_number,
            timestamp_ms=frame.timestamp_ms,
            detections=all_detections,
            state_changes=state_changes,
            processing_time_ms=processing_time_ms,
        )

    def _calculate_state_changes(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate what changed between states.

        Args:
            previous: Previous state
            current: Current state

        Returns:
            Dictionary of changed fields with {old, new} values
        """
        changes = {}
        self._compare_dicts(previous, current, [], changes)
        return changes

    def _compare_dicts(
        self,
        prev: dict[str, Any],
        curr: dict[str, Any],
        path: list[str],
        changes: dict[str, Any],
    ) -> None:
        """Recursively compare dictionaries."""
        all_keys = set(prev.keys()) | set(curr.keys())

        for key in all_keys:
            current_path = path + [key]
            path_str = ".".join(current_path)

            prev_val = prev.get(key)
            curr_val = curr.get(key)

            if isinstance(prev_val, dict) and isinstance(curr_val, dict):
                self._compare_dicts(prev_val, curr_val, current_path, changes)
            elif prev_val != curr_val:
                # Handle state value objects with "value" key
                if isinstance(prev_val, dict) and "value" in prev_val:
                    prev_val = prev_val.get("value")
                if isinstance(curr_val, dict) and "value" in curr_val:
                    curr_val = curr_val.get("value")

                if prev_val != curr_val:
                    changes[path_str] = {
                        "old": prev_val,
                        "new": curr_val,
                    }

    def reset(self) -> None:
        """Reset frame counter."""
        self.frame_number = 0
