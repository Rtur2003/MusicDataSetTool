"""
Retry utilities for handling transient failures in API calls.
"""

import time
import functools
from typing import Callable, Type, Tuple, Optional
from .logger import get_logger
from .exceptions import APIError

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (APIError, ConnectionError, TimeoutError)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def fetch_data():
            # API call that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        
            raise last_exception
            
        return wrapper
    return decorator


def retry_on_rate_limit(
    max_retries: int = 3,
    base_delay: float = 60.0
) -> Callable:
    """
    Decorator specifically for handling rate limit errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds when rate limited
        
    Returns:
        Decorated function with rate limit retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    is_rate_limit = any(
                        phrase in error_msg 
                        for phrase in ['rate limit', 'too many requests', '429']
                    )
                    
                    if is_rate_limit and attempt < max_retries:
                        delay = base_delay * (attempt + 1)
                        logger.warning(
                            f"Rate limit hit for {func.__name__}. "
                            f"Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}..."
                        )
                        time.sleep(delay)
                    else:
                        raise
                        
            raise APIError(f"Rate limit exceeded after {max_retries} retries")
            
        return wrapper
    return decorator
