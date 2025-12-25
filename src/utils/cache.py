"""
Caching utilities for storing API responses and expensive computations.
"""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import functools

from .logger import get_logger

logger = get_logger(__name__)


class SimpleCache:
    """Simple file-based cache for API responses and computations."""
    
    def __init__(self, cache_dir: str = '.cache', ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)
        logger.debug(f"Cache initialized: {self.cache_dir} (TTL: {ttl_seconds}s)")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            timestamp = cached_data['timestamp']
            value = cached_data['value']
            
            # Check if expired
            age = datetime.now() - timestamp
            if age > self.ttl:
                logger.debug(f"Cache expired: {key} (age: {age.total_seconds():.0f}s)")
                cache_path.unlink()
                return None
            
            logger.debug(f"Cache hit: {key}")
            return value
            
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        try:
            cached_data = {
                'timestamp': datetime.now(),
                'value': value
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.debug(f"Cached: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Deleted from cache: {key}")
            return True
        
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob('*.cache'):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of expired entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob('*.cache'):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                timestamp = cached_data['timestamp']
                age = datetime.now() - timestamp
                
                if age > self.ttl:
                    cache_file.unlink()
                    count += 1
                    
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {str(e)}")
                cache_file.unlink()
                count += 1
        
        if count > 0:
            logger.info(f"Cleared {count} expired cache entries")
        
        return count


def cached(ttl_seconds: int = 3600, cache_dir: str = '.cache'):
    """
    Decorator for caching function results.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        cache_dir: Directory to store cache files
        
    Returns:
        Decorated function with caching
        
    Example:
        @cached(ttl_seconds=3600)
        def expensive_operation(param):
            # Long-running computation
            return result
    """
    cache = SimpleCache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        # Add cache management methods to the decorated function
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_clear_expired = lambda: cache.clear_expired()
        
        return wrapper
    
    return decorator
