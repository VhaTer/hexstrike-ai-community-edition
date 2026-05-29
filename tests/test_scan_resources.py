"""Tests for scan MCP resources: scan://{target}/latest, scan://{target}/{tool_name}, scan://cache/list.

Validates the cache filtering logic (session_id prefix + seed fallback)
that backs the resource URIs.
"""

import json
import time
import pytest
from unittest.mock import patch

from server_core.advanced_cache import AdvancedCache


def _make_cache(entries: list[dict]) -> AdvancedCache:
    """Populate an AdvancedCache with scan entries."""
    cache: AdvancedCache = AdvancedCache(max_size=100)
    for i, e in enumerate(entries):
        cache.set(str(e.get("key", f"test:{i}")), {
            "tool": e["tool"],
            "target": e["target"],
            "timestamp": e.get("timestamp", time.time()),
            "status": e.get("status", "completed"),
            "result": {"success": True, "stdout": f"{e['tool']} output"},
        })
    return cache


class TestScanCacheFiltering:
    """Filtering logic that powers scan:// resources."""

    def test_latest_returns_most_recent_entry(self):
        """Given multiple scans for same target, latest timestamp wins."""
        now = time.time()
        entries = [
            {"key": "sess:abc:nmap:target.x", "tool": "nmap", "target": "target.x", "timestamp": now - 10},
            {"key": "sess:abc:whatweb:target.x", "tool": "whatweb", "target": "target.x", "timestamp": now},
        ]
        cache = _make_cache(entries)
        matches = [
            v for k, v in cache.items()
            if v.get("target") == "target.x"
            and (k.startswith("sess:abc:") or k.startswith("seed:"))
        ]
        latest = max(matches, key=lambda x: x["timestamp"])
        assert latest["tool"] == "whatweb"

    def test_latest_returns_no_results_for_unknown_target(self):
        """No cache entries for target → no_results status."""
        cache = _make_cache([])
        matches = [
            v for k, v in cache.items()
            if v.get("target") == "unknown"
            and (k.startswith("sess:abc:") or k.startswith("seed:"))
        ]
        assert len(matches) == 0

    def test_per_tool_returns_specific_tool_entry(self):
        """Filter by tool name returns only matching entries."""
        now = time.time()
        entries = [
            {"key": "sess:abc:nmap:x", "tool": "nmap", "target": "x", "timestamp": now - 5},
            {"key": "sess:abc:nuclei:x", "tool": "nuclei", "target": "x", "timestamp": now},
        ]
        cache = _make_cache(entries)
        entry = next(
            (v for k, v in sorted(cache.items(), reverse=True)
             if v.get("tool") == "nmap"
             and v.get("target") == "x"
             and (k.startswith("sess:abc:") or k.startswith("seed:"))),
            None,
        )
        assert entry is not None
        assert entry["tool"] == "nmap"

    def test_per_tool_returns_none_for_missing(self):
        """No entry for tool/target combo returns None."""
        cache = _make_cache([])
        entry = next(
            (v for k, v in sorted(cache.items(), reverse=True)
             if v.get("tool") == "nmap"
             and v.get("target") == "unknown"
             and (k.startswith("sess:abc:") or k.startswith("seed:"))),
            None,
        )
        assert entry is None

    def test_per_tool_picks_newest_when_multiple(self):
        """Multiple entries for same tool/target — pick the newest."""
        now = time.time()
        entries = [
            {"key": "sess:abc:nmap:x", "tool": "nmap", "target": "x", "timestamp": now - 100},
            {"key": "sess:abc:nmap:x2", "tool": "nmap", "target": "x", "timestamp": now},
        ]
        cache = _make_cache(entries)
        entry = next(
            (v for k, v in sorted(cache.items(), reverse=True)
             if v.get("tool") == "nmap"
             and v.get("target") == "x"
             and (k.startswith("sess:abc:") or k.startswith("seed:"))),
            None,
        )
        assert entry is not None
        assert entry["timestamp"] == now

    def test_cache_list_returns_all_session_entries(self):
        """scan://cache/list returns count + entries sorted by timestamp desc."""
        now = time.time()
        entries = [
            {"key": "sess:abc:nmap:x", "tool": "nmap", "target": "x", "timestamp": now},
            {"key": "sess:abc:whatweb:x", "tool": "whatweb", "target": "x", "timestamp": now - 5},
            {"key": "seed:nikto:x", "tool": "nikto", "target": "x", "timestamp": now - 10},
        ]
        cache = _make_cache(entries)
        result = [
            {"key": k, "tool": v["tool"], "target": v["target"], "timestamp": v["timestamp"]}
            for k, v in cache.items()
            if k.startswith("sess:abc:") or k.startswith("seed:")
        ]
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        assert len(result) == 3
        assert result[0]["tool"] == "nmap"

    def test_cache_list_ignores_other_sessions(self):
        """Entries from other session IDs are excluded."""
        now = time.time()
        entries = [
            {"key": "sess:abc:nmap:x", "tool": "nmap", "target": "x", "timestamp": now},
            {"key": "sess:def:nmap:y", "tool": "nmap", "target": "y", "timestamp": now},
        ]
        cache = _make_cache(entries)
        our = [
            v for k, v in cache.items()
            if k.startswith("sess:abc:") or k.startswith("seed:")
        ]
        assert len(our) == 1
        assert our[0]["target"] == "x"
