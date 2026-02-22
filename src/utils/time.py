"""Time utilities for game-study."""

import time
from datetime import datetime


def get_timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds.

    Returns:
        Timestamp in milliseconds since Unix epoch
    """
    return int(time.time() * 1000)


def format_timestamp(ts_ms: int, format_str: str | None = None) -> str:
    """
    Format timestamp to string.

    Args:
        ts_ms: Timestamp in milliseconds
        format_str: Optional format string (default: ISO format)

    Returns:
        Formatted timestamp string
    """
    dt = datetime.fromtimestamp(ts_ms / 1000)
    if format_str:
        return dt.strftime(format_str)
    return dt.isoformat()


def get_elapsed_ms(start_ts_ms: int) -> int:
    """
    Get elapsed time in milliseconds.

    Args:
        start_ts_ms: Start timestamp in milliseconds

    Returns:
        Elapsed time in milliseconds
    """
    return get_timestamp_ms() - start_ts_ms
