"""Speech queue manager for voice responses."""

import queue
import threading
import time
from dataclasses import dataclass, field


@dataclass
class QueuedSpeech:
    """Speech item in queue."""

    text: str
    priority: int = 2
    trigger_id: str | None = None
    audio_data: bytes | None = None
    duration_ms: int | None = None
    queued_at: float = field(default_factory=time.time)

    @property
    def wait_time_ms(self) -> int:
        """Time spent waiting in queue."""
        return int((time.time() - self.queued_at) * 1000)


class SpeechQueue:
    """
    Priority queue for speech responses.

    Manages a queue of speech items with priority-based ordering
    and interruption support.
    """

    def __init__(self, max_size: int = 10):
        """
        Initialize speech queue.

        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue: queue.Queue[QueuedSpeech] = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._current: QueuedSpeech | None = None

    def enqueue(self, speech: QueuedSpeech) -> bool:
        """
        Add speech to queue.

        Args:
            speech: Speech item to add

        Returns:
            True if added, False if queue full
        """
        try:
            self._queue.put_nowait(speech)
            return True
        except queue.Full:
            # Try to remove lowest priority item
            with self._lock:
                items = list(self._queue.queue)
                if items:
                    lowest = max(items, key=lambda s: s.priority)
                    if speech.priority < lowest.priority:
                        items.remove(lowest)
                        self._queue = queue.Queue(maxsize=self.max_size)
                        for item in items:
                            self._queue.put_nowait(item)
                        self._queue.put_nowait(speech)
                        return True
            return False

    def dequeue(self, timeout: float = 0.1) -> QueuedSpeech | None:
        """
        Get next speech from queue.

        Args:
            timeout: Maximum time to wait

        Returns:
            Next speech item or None if empty
        """
        try:
            speech = self._queue.get(timeout=timeout)
            self._current = speech
            return speech
        except queue.Empty:
            return None

    def peek(self) -> QueuedSpeech | None:
        """
        Peek at next speech without removing.

        Returns:
            Next speech item or None if empty
        """
        try:
            return self._queue.queue[0]
        except (AttributeError, IndexError):
            return None

    def clear(self) -> int:
        """
        Clear the queue.

        Returns:
            Number of items cleared
        """
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        self._current = None
        return count

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    @property
    def current(self) -> QueuedSpeech | None:
        """Get currently playing speech."""
        return self._current

    def get_highest_priority(self) -> int | None:
        """Get highest priority in queue."""
        try:
            items = list(self._queue.queue)
            if items:
                return min(s.priority for s in items)
        except (AttributeError, IndexError):
            pass
        return None

    def has_priority_higher_than(self, priority: int) -> bool:
        """Check if queue has higher priority item.

        Args:
            priority: Priority to compare against

        Returns:
            True if higher priority item exists
        """
        highest = self.get_highest_priority()
        return highest is not None and highest < priority
