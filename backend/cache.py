"""
Simple in-memory cache with TTL (time-to-live) support.
Helps reduce Yahoo Finance API rate limiting by caching responses.
"""

import time
from typing import Any, Optional, Dict, Tuple
from datetime import datetime


class SimpleCache:
    """In-memory cache with automatic expiration."""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key in self._cache:
            data, expiry = self._cache[key]
            if time.time() < expiry:
                self._hits += 1
                print(f"[CACHE HIT] {key}")
                return data
            else:
                # Expired, remove from cache
                del self._cache[key]
                self._misses += 1
                print(f"[CACHE MISS - EXPIRED] {key}")
        else:
            self._misses += 1
            print(f"[CACHE MISS] {key}")
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int):
        """
        Store value in cache with expiration time.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
        """
        expiry = time.time() + ttl_seconds
        self._cache[key] = (value, expiry)
        print(f"[CACHE SET] {key} (TTL: {ttl_seconds}s)")
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        print("[CACHE CLEARED]")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def cleanup_expired(self):
        """Remove all expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self._cache.items()
            if current_time >= expiry
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            print(f"[CACHE CLEANUP] Removed {len(expired_keys)} expired entries")


# Global cache instance
cache = SimpleCache()
