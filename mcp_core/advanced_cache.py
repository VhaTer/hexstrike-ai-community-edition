"""
mcp_core/advanced_cache.py

AdvancedCache — LRU + TTL-adaptive scan result cache for HexStrike AI-PULSE.

Replaces the bare _scan_cache dict in server_setup.py with:
  - Capacity-bounded storage (default 500 entries)
  - Per-entry TTL with adaptive extension for costly scans
  - LRU eviction when capacity is reached
  - Thread-safe via threading.Lock (called from executor threads)
  - Hit/miss statistics

Drop-in compatible: same key format as _scan_cache ("tool:target"),
same entry structure {"tool", "target", "result", "timestamp"}.

Usage:
    from mcp_core.advanced_cache import AdvancedCache
    _scan_cache = AdvancedCache()

    # Write
    _scan_cache.set("nmap:10.10.10.10", {
        "tool": "nmap", "target": "10.10.10.10",
        "result": {...}, "timestamp": time.time(),
    }, execution_time=45.0)

    # Read
    entry = _scan_cache.get("nmap:10.10.10.10")  # None if miss or expired

    # Stats
    stats = _scan_cache.stats()
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------
_TTL_DEFAULT  = 1800   # 30 min — fast scans
_TTL_MEDIUM   = 3600   # 60 min — scans > 10s
_TTL_LONG     = 5400   # 90 min — scans > 60s
_TTL_MAX      = 7200   # 2h    — absolute ceiling
_CLEANUP_INTERVAL = 120  # run expiry sweep every 2 min


class AdvancedCache:
    """
    LRU + TTL-adaptive in-memory cache for scan results.

    Thread-safe. Designed to be used as a module-level singleton
    replacing the bare _scan_cache dict in server_setup.py.
    """

    def __init__(self, capacity: int = 500, default_ttl: int = _TTL_DEFAULT) -> None:
        self._capacity    = capacity
        self._default_ttl = default_ttl
        # OrderedDict preserves insertion order for LRU tracking
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock        = threading.Lock()
        self._last_cleanup = time.time()

        # Stats
        self._hits   = 0
        self._misses = 0
        self._evictions    = 0
        self._expirations  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Return the cached entry for key, or None if missing/expired.
        Moves the entry to the end (most recently used) on hit.
        """
        self._maybe_cleanup()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # TTL check
            if time.time() - entry["_cached_at"] > entry["_ttl"]:
                del self._store[key]
                self._expirations += 1
                self._misses += 1
                return None

            # LRU: move to end
            self._store.move_to_end(key)
            self._hits += 1
            entry["_access_count"] = entry.get("_access_count", 0) + 1
            return entry["data"]

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        execution_time: float = 0.0,
    ) -> None:
        """
        Store value under key.

        TTL is adaptive based on execution_time:
          < 10s  → 30 min
          10–60s → 60 min
          > 60s  → 90 min
        """
        ttl = self._adaptive_ttl(execution_time)
        record = {
            "data":          value,
            "_cached_at":    time.time(),
            "_ttl":          ttl,
            "_exec_time":    execution_time,
            "_access_count": 0,
        }
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = record

            # Evict LRU entries if over capacity
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Flush all entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False
        return time.time() - entry["_cached_at"] <= entry["_ttl"]

    def keys(self):
        """Return a snapshot of current (non-expired) keys."""
        now = time.time()
        with self._lock:
            return [
                k for k, v in self._store.items()
                if now - v["_cached_at"] <= v["_ttl"]
            ]

    def items(self):
        """Return a snapshot of (key, data) for non-expired entries."""
        now = time.time()
        with self._lock:
            return [
                (k, v["data"]) for k, v in self._store.items()
                if now - v["_cached_at"] <= v["_ttl"]
            ]

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = round(self._hits / total, 3) if total > 0 else 0.0
        with self._lock:
            size = len(self._store)
        return {
            "hits":        self._hits,
            "misses":      self._misses,
            "hit_rate":    hit_rate,
            "size":        size,
            "capacity":    self._capacity,
            "evictions":   self._evictions,
            "expirations": self._expirations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adaptive_ttl(execution_time: float) -> int:
        """Compute TTL based on how expensive the scan was."""
        if execution_time > 60:
            return min(_TTL_LONG, _TTL_MAX)
        if execution_time > 10:
            return min(_TTL_MEDIUM, _TTL_MAX)
        return _TTL_DEFAULT

    def _maybe_cleanup(self) -> None:
        """Periodically sweep and remove expired entries."""
        now = time.time()
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        expired = []
        with self._lock:
            for k, v in list(self._store.items()):
                if now - v["_cached_at"] > v["_ttl"]:
                    expired.append(k)
            for k in expired:
                del self._store[k]
                self._expirations += 1
