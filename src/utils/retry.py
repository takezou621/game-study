"""Retry functionality for API calls."""

import asyncio
import time
import logging
from typing import Callable, Type, Tuple, Optional, Any, Union
import random

from .exceptions import APIError, RateLimitError

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        APIError,
        RateLimitError,
        ConnectionError,
        TimeoutError,
    ),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to avoid thundering herd
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry (attempt, error, delay)

    Returns:
        Result from the function

    Raises:
        The last exception if all retries are exhausted

    Example:
        async def fetch_data():
            return await api_call()

        result = await retry_with_backoff(
            fetch_data,
            max_retries=3,
            base_delay=1.0
        )
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()

        except Exception as e:
            last_exception = e

            # Check if this is a retryable exception
            is_retryable = isinstance(e, retryable_exceptions)

            # Special handling for RateLimitError
            if isinstance(e, RateLimitError) and e.retry_after is not None:
                delay = min(e.retry_after, max_delay)
                logger.info(
                    f"Rate limit hit, using retry_after={delay:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            elif is_retryable:
                # Calculate delay with exponential backoff
                delay = min(
                    base_delay * (exponential_base ** attempt),
                    max_delay
                )

                # Add jitter if enabled
                if jitter:
                    delay = delay * (0.5 + random.random() * 0.5)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
            else:
                # Non-retryable exception, raise immediately
                logger.error(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            # Call on_retry callback if provided
            if on_retry:
                on_retry(attempt + 1, e, delay)

            # If we've exhausted retries, raise
            if attempt >= max_retries:
                logger.error(
                    f"All {max_retries} retry attempts exhausted. "
                    f"Last error: {type(last_exception).__name__}: {last_exception}"
                )
                raise last_exception

            # Wait before retrying
            await asyncio.sleep(delay)


def retry_with_backoff_sync(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        APIError,
        RateLimitError,
        ConnectionError,
        TimeoutError,
    ),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """
    Retry a synchronous function with exponential backoff.

    Args:
        func: Synchronous function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to avoid thundering herd
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry (attempt, error, delay)

    Returns:
        Result from the function

    Raises:
        The last exception if all retries are exhausted

    Example:
        def fetch_data():
            return api_call()

        result = retry_with_backoff_sync(
            fetch_data,
            max_retries=3,
            base_delay=1.0
        )
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()

        except Exception as e:
            last_exception = e

            # Check if this is a retryable exception
            is_retryable = isinstance(e, retryable_exceptions)

            # Special handling for RateLimitError
            if isinstance(e, RateLimitError) and e.retry_after is not None:
                delay = min(e.retry_after, max_delay)
                logger.info(
                    f"Rate limit hit, using retry_after={delay:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            elif is_retryable:
                # Calculate delay with exponential backoff
                delay = min(
                    base_delay * (exponential_base ** attempt),
                    max_delay
                )

                # Add jitter if enabled
                if jitter:
                    delay = delay * (0.5 + random.random() * 0.5)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
            else:
                # Non-retryable exception, raise immediately
                logger.error(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            # Call on_retry callback if provided
            if on_retry:
                on_retry(attempt + 1, e, delay)

            # If we've exhausted retries, raise
            if attempt >= max_retries:
                logger.error(
                    f"All {max_retries} retry attempts exhausted. "
                    f"Last error: {type(last_exception).__name__}: {last_exception}"
                )
                raise last_exception

            # Wait before retrying
            time.sleep(delay)
