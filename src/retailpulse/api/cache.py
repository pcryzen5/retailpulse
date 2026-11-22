"""In-process thread-safe TTL cache for API responses and metadata.

Implements API §1.1 caching requirements:
- In-process caching for KPI overview and champion model metadata (default 60s TTL).
- Thread-safe get/set/clear operations.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    """Thread-safe in-memory cache with time-to-live expiration."""

    def __init__(self, default_ttl_seconds: float = 60.0) -> None:
        self._default_ttl = default_ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value if present and not expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """Store a value with an expiration timestamp."""
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        expires_at = time.monotonic() + ttl
        with self._lock:
            self._data[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        """Remove a key from the cache if present."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._data.clear()


# Default application-wide TTL cache instance
api_cache = TTLCache(default_ttl_seconds=60.0)
