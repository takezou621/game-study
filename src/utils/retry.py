"""Retry utilities with exponential backoff.

Provides decorators and utilities for retrying operations
with configurable backoff strategies.
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None
):
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to delays
        exceptions: Tuple of exception types to retry on
        on_retry: Callback called on each retry (exception, attempt, delay)

    Returns:
        Decorated function

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )

                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], Any] | None = None
):
    """
    Async decorator for retrying a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to delays
        exceptions: Tuple of exception types to retry on
        on_retry: Async callback called on each retry

    Returns:
        Decorated async function

    Example:
        @async_retry_with_backoff(max_retries=3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                return await session.get("https://api.example.com/data")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {e}"
                    )

                    if on_retry:
                        result = on_retry(e, attempt + 1, delay)
                        if asyncio.iscoroutine(result):
                            await result

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry logic with manual control.

    Example:
        with RetryContext(max_retries=3) as retry:
            while retry.should_retry():
                try:
                    result = risky_operation()
                    break
                except Exception as e:
                    retry.record_failure(e)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry context.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation
            jitter: Add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

        self.attempt = 0
        self.last_exception: Exception | None = None

    def __enter__(self) -> 'RetryContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def should_retry(self) -> bool:
        """Check if we should attempt another retry."""
        return self.attempt <= self.max_retries

    def record_failure(self, exception: Exception) -> None:
        """
        Record a failure and wait before next attempt.

        Args:
            exception: The exception that occurred

        Raises:
            Exception: If max retries exceeded
        """
        self.last_exception = exception
        self.attempt += 1

        if self.attempt > self.max_retries:
            logger.error(f"All {self.max_retries} retry attempts exceeded")
            raise exception

        # Calculate delay
        delay = min(
            self.base_delay * (self.exponential_base ** (self.attempt - 1)),
            self.max_delay
        )

        if self.jitter:
            delay = delay * (0.5 + random.random())

        logger.warning(
            f"Retry {self.attempt}/{self.max_retries} after {delay:.2f}s"
        )

        time.sleep(delay)

    def get_delay(self) -> float:
        """Get the delay for the next retry without sleeping."""
        delay = min(
            self.base_delay * (self.exponential_base ** self.attempt),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay
