"""
Islamic AI Assistant — Cache Utilities

TTL-based cache untuk embedding dan RAG results.
Menggunakan cachetools.TTLCache dengan thread-safe operations.
"""

import hashlib
import threading
from typing import Any, Callable, Optional, TypeVar
from cachetools import TTLCache

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class CacheManager:
    """
    Manager untuk berbagai cache instances dengan TTL.
    Thread-safe dengan lock per cache.
    """

    def __init__(self):
        """Initialize cache instances dengan konfigurasi yang berbeda"""

        # Cache untuk embedding results
        # maxsize=1000, TTL=3600 detik (1 jam)
        self.embedding_cache = TTLCache(maxsize=1000, ttl=3600)
        self._embedding_lock = threading.Lock()

        # Cache untuk RAG retrieve results
        # maxsize=500, TTL=1800 detik (30 menit)
        self.rag_cache = TTLCache(maxsize=500, ttl=1800)
        self._rag_lock = threading.Lock()

        # Cache untuk intent classification results
        # maxsize=2000, TTL=7200 detik (2 jam)
        self.intent_cache = TTLCache(maxsize=2000, ttl=7200)
        self._intent_lock = threading.Lock()

        logger.info("Cache manager initialized with 3 cache instances")

    def get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key dari arguments dengan hashing.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hash string sebagai cache key
        """
        # Combine args dan kwargs menjadi satu string
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)

        # Hash dengan SHA256 untuk key yang konsisten
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_or_compute(
        self,
        cache: TTLCache,
        lock: threading.Lock,
        key: str,
        compute_fn: Callable[[], T],
        force_refresh: bool = False
    ) -> T:
        """
        Get value dari cache atau compute jika tidak ada.
        Thread-safe operation dengan lock.

        Args:
            cache: TTLCache instance yang akan digunakan
            lock: Threading lock untuk thread safety
            key: Cache key
            compute_fn: Function untuk compute value jika cache miss
            force_refresh: Paksa refresh cache (skip cache lookup)

        Returns:
            Cached value atau hasil dari compute_fn
        """
        # Check cache first (tanpa lock untuk read)
        if not force_refresh:
            with lock:
                if key in cache:
                    logger.debug(f"Cache HIT: {key[:16]}...")
                    return cache[key]

        # Cache miss - compute value
        logger.debug(f"Cache MISS: {key[:16]}...")
        value = compute_fn()

        # Store ke cache (dengan lock)
        with lock:
            cache[key] = value

        return value

    def invalidate(self, cache_name: str, key: Optional[str] = None):
        """
        Invalidate cache entry atau seluruh cache.

        Args:
            cache_name: Nama cache ('embedding', 'rag', 'intent')
            key: Specific key to invalidate (None = clear all)
        """
        cache_map = {
            'embedding': (self.embedding_cache, self._embedding_lock),
            'rag': (self.rag_cache, self._rag_lock),
            'intent': (self.intent_cache, self._intent_lock)
        }

        if cache_name not in cache_map:
            logger.warning(f"Unknown cache name: {cache_name}")
            return

        cache, lock = cache_map[cache_name]

        with lock:
            if key is None:
                cache.clear()
                logger.info(f"Cleared all entries in {cache_name} cache")
            elif key in cache:
                del cache[key]
                logger.info(f"Invalidated key in {cache_name} cache: {key[:16]}...")

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get cache statistics untuk monitoring.

        Returns:
            Dictionary dengan stats per cache
        """
        stats = {}

        with self._embedding_lock:
            stats['embedding'] = {
                'size': len(self.embedding_cache),
                'maxsize': self.embedding_cache.maxsize,
                'ttl': self.embedding_cache.ttl
            }

        with self._rag_lock:
            stats['rag'] = {
                'size': len(self.rag_cache),
                'maxsize': self.rag_cache.maxsize,
                'ttl': self.rag_cache.ttl
            }

        with self._intent_lock:
            stats['intent'] = {
                'size': len(self.intent_cache),
                'maxsize': self.intent_cache.maxsize,
                'ttl': self.intent_cache.ttl
            }

        return stats

# Global singleton instance
_cache_manager: Optional[CacheManager] = None
_cache_manager_lock = threading.Lock()

def get_cache_manager() -> CacheManager:
    """
    Get singleton instance dari CacheManager.
    Thread-safe initialization.

    Returns:
        CacheManager instance

    Example:
        >>> from src.utils.cache import get_cache_manager
        >>> cache = get_cache_manager()
        >>> key = cache.get_cache_key("query", model="llama")
        >>> result = cache.get_or_compute(
        ...     cache.embedding_cache,
        ...     cache._embedding_lock,
        ...     key,
        ...     lambda: expensive_compute()
        ... )
    """
    global _cache_manager

    if _cache_manager is None:
        with _cache_manager_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager()

    return _cache_manager
