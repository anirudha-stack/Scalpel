"""Retry utilities for handling transient failures."""

import time
import logging
from typing import TypeVar, Callable, Optional, Tuple
from functools import wraps

T = TypeVar("T")
logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[type, ...] = (Exception,),
):
    """Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper

    return decorator


class RetryHandler:
    """Handler for retry logic with callbacks."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        exceptions: Tuple[type, ...] = (Exception,),
    ) -> None:
        """Initialize the retry handler.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries
            on_retry: Optional callback called on each retry
            exceptions: Tuple of exception types to catch
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.on_retry = on_retry
        self.exceptions = exceptions

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)

                    if self.on_retry:
                        self.on_retry(attempt, e)
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                    time.sleep(delay)

        raise last_exception  # type: ignore

    async def execute_async(
        self, func: Callable[..., T], *args, **kwargs
    ) -> T:
        """Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            The last exception if all retries fail
        """
        import asyncio

        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)

                    if self.on_retry:
                        self.on_retry(attempt, e)

                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore
