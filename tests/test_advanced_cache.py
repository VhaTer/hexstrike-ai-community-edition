import threading
import time

import pytest
from unittest.mock import patch, MagicMock

from server_core.advanced_cache import AdvancedCache
from mcp_core.server_setup import _ScanCache, _cache_key_for, _collect_cached_scans


# =========================================================================
# AdvancedCache — core LRU + TTL cache
# =========================================================================

class TestAdvancedCacheInit:
    def test_default_params(self):
        c = AdvancedCache()
        assert c.max_size == 1000
        assert c.default_ttl == 3600
        assert c.cache == {}
        assert c.access_times == {}
        assert c.ttl_times == {}
        assert isinstance(c.cache_lock, type(threading.RLock()))
        assert c.hit_count == 0
        assert c.miss_count == 0
        assert c._cleanup_thread_started is False

    def test_custom_params(self):
        c = AdvancedCache(max_size=50, default_ttl=300)
        assert c.max_size == 50
        assert c.default_ttl == 300


class TestAdvancedCacheSet:
    @patch("threading.Thread")
    def test_set_starts_cleanup_thread(self, mock_thread):
        c = AdvancedCache()
        assert c._cleanup_thread_started is False
        c.set("k1", "v1")
        assert c._cleanup_thread_started is True
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch("threading.Thread")
    def test_set_only_starts_cleanup_once(self, mock_thread):
        c = AdvancedCache()
        c.set("k1", "v1")
        c.set("k2", "v2")
        assert mock_thread.call_count == 1

    @patch("threading.Thread")
    def test_set_default_ttl(self, mock_thread):
        c = AdvancedCache(default_ttl=999)
        c.set("k", "v")
        entry_time = c.ttl_times["k"]
        expected = time.time() + 999
        assert abs(entry_time - expected) < 1

    @patch("threading.Thread")
    def test_set_custom_ttl(self, mock_thread):
        c = AdvancedCache(default_ttl=999)
        c.set("k", "v", ttl=60)
        entry_time = c.ttl_times["k"]
        expected = time.time() + 60
        assert abs(entry_time - expected) < 1

    @patch("threading.Thread")
    def test_set_stores_value_and_access_time(self, mock_thread):
        c = AdvancedCache()
        now = time.time()
        c.set("k", "value123")
        assert c.cache["k"] == "value123"
        assert abs(c.access_times["k"] - now) < 1
        assert c.ttl_times["k"] > now

    @patch("threading.Thread")
    def test_set_evicts_lru_when_full(self, mock_thread):
        c = AdvancedCache(max_size=3)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.get("a")
        c.set("d", 4)
        assert "b" not in c.cache
        assert "a" in c.cache
        assert "c" in c.cache
        assert "d" in c.cache
        assert len(c.cache) == 3

    @patch("threading.Thread")
    def test_set_no_eviction_when_key_already_present(self, mock_thread):
        c = AdvancedCache(max_size=1)
        c.set("a", 1)
        c.set("a", 2)
        assert c.cache["a"] == 2
        assert len(c.cache) == 1


class TestAdvancedCacheGet:
    def test_get_miss_nonexistent(self):
        c = AdvancedCache()
        val = c.get("nonexistent")
        assert val is None
        assert c.miss_count == 1

    @patch("threading.Thread")
    def test_get_hit(self, mock_thread):
        c = AdvancedCache()
        c.set("k", "stored")
        val = c.get("k")
        assert val == "stored"
        assert c.hit_count == 1

    @patch("threading.Thread")
    def test_get_expired_removes_entry(self, mock_thread):
        c = AdvancedCache()
        c.set("k", "stored", ttl=0)
        val = c.get("k")
        assert val is None
        assert "k" not in c.cache
        assert "k" not in c.access_times
        assert "k" not in c.ttl_times
        assert c.miss_count == 1

    @patch("threading.Thread")
    def test_get_hit_updates_access_time(self, mock_thread):
        c = AdvancedCache()
        c.set("k", "v")
        with patch("time.time", return_value=99999.0):
            c.get("k")
        assert c.access_times["k"] == 99999.0

    @patch("threading.Thread")
    def test_get_hit_no_ttl_entry(self, mock_thread):
        """Key in cache but not in ttl_times — should still be a hit."""
        c = AdvancedCache()
        c.cache["k"] = "val"
        c.access_times["k"] = time.time()
        val = c.get("k")
        assert val == "val"
        assert c.hit_count == 1


class TestAdvancedCacheDelete:
    def test_delete_existing(self):
        c = AdvancedCache()
        c.cache["k"] = "v"
        c.access_times["k"] = time.time()
        c.ttl_times["k"] = time.time() + 100
        result = c.delete("k")
        assert result is True
        assert "k" not in c.cache
        assert "k" not in c.access_times
        assert "k" not in c.ttl_times

    def test_delete_missing(self):
        c = AdvancedCache()
        result = c.delete("nonexistent")
        assert result is False


class TestAdvancedCacheClear:
    def test_clear_empties_all_dicts(self):
        c = AdvancedCache()
        c.cache["a"] = 1
        c.access_times["a"] = time.time()
        c.ttl_times["a"] = time.time() + 100
        c.clear()
        assert c.cache == {}
        assert c.access_times == {}
        assert c.ttl_times == {}


class TestAdvancedCacheLen:
    def test_len(self):
        c = AdvancedCache()
        assert len(c) == 0
        c.cache["a"] = 1
        assert len(c) == 1


class TestAdvancedCacheItemsKeysValues:
    @patch("threading.Thread")
    def test_items(self, mock_thread):
        c = AdvancedCache()
        c.set("a", 1)
        c.set("b", 2)
        items = c.items()
        assert len(items) == 2
        assert ("a", 1) in items
        assert ("b", 2) in items

    @patch("threading.Thread")
    def test_keys(self, mock_thread):
        c = AdvancedCache()
        c.set("a", 1)
        items = c.keys()
        assert "a" in items

    @patch("threading.Thread")
    def test_values(self, mock_thread):
        c = AdvancedCache()
        c.set("a", 1)
        items = c.values()
        assert 1 in items


class TestAdvancedCacheEvictLru:
    def test_evict_lru_empty(self):
        c = AdvancedCache()
        c._evict_lru()

    @patch("threading.Thread")
    def test_evict_lru_removes_oldest(self, mock_thread):
        c = AdvancedCache()
        c.cache["old"] = "o"
        c.access_times["old"] = 100.0
        c.ttl_times["old"] = 999.0
        c.cache["mid"] = "m"
        c.access_times["mid"] = 200.0
        c.ttl_times["mid"] = 999.0
        c.cache["new"] = "n"
        c.access_times["new"] = 300.0
        c.ttl_times["new"] = 999.0
        c._evict_lru()
        assert "old" not in c.cache
        assert "mid" in c.cache
        assert "new" in c.cache


class TestAdvancedCacheStats:
    def test_get_stats_no_requests(self):
        c = AdvancedCache()
        stats = c.get_stats()
        assert stats["hit_rate"] == 0
        assert stats["size"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0

    def test_get_stats_with_hits_and_misses(self):
        c = AdvancedCache()
        c.cache["a"] = 1
        c.hit_count = 80
        c.miss_count = 20
        stats = c.get_stats()
        assert stats["size"] == 1
        assert stats["hit_rate"] == 80.0
        assert stats["utilization"] == 0.1

    def test_get_stats_only_misses(self):
        c = AdvancedCache()
        c.miss_count = 10
        stats = c.get_stats()
        assert stats["hit_rate"] == 0.0

    def test_get_stats_only_hits(self):
        c = AdvancedCache()
        c.hit_count = 5
        stats = c.get_stats()
        assert stats["hit_rate"] == 100.0


class TestAdvancedCacheCleanupExpired:
    """Test _cleanup_expired by running one full iteration then breaking
    with KeyboardInterrupt (BaseException, not caught by except Exception)."""

    def _run_one_iteration(self, c):
        """Let time.sleep() succeed once (cleanup logic runs), then
        raise KeyboardInterrupt on second call to break the infinite loop."""
        calls = []
        def sleeper(*a, **kw):
            calls.append(1)
            if len(calls) >= 2:
                raise KeyboardInterrupt()
        with patch("time.sleep", side_effect=sleeper):
            try:
                c._cleanup_expired()
            except KeyboardInterrupt:
                pass

    @patch("threading.Thread")
    def test_cleanup_removes_expired_preserves_fresh(self, mock_thread):
        now = time.time()
        c = AdvancedCache()
        c.cache["fresh"] = "ok"
        c.access_times["fresh"] = now
        c.ttl_times["fresh"] = now + 3600
        c.cache["stale"] = "gone"
        c.access_times["stale"] = now - 3600
        c.ttl_times["stale"] = now - 60
        self._run_one_iteration(c)
        assert "fresh" in c.cache
        assert "stale" not in c.cache

    @patch("threading.Thread")
    def test_cleanup_no_expired_keys_no_log(self, mock_thread):
        now = time.time()
        c = AdvancedCache()
        c.cache["a"] = 1
        c.access_times["a"] = now
        c.ttl_times["a"] = now + 3600
        with patch("logging.Logger.debug") as mock_log:
            self._run_one_iteration(c)
        mock_log.assert_not_called()

    @patch("threading.Thread")
    def test_cleanup_with_expired_keys_logs(self, mock_thread):
        now = time.time()
        c = AdvancedCache()
        c.cache["stale"] = "gone"
        c.access_times["stale"] = now - 3600
        c.ttl_times["stale"] = now - 60
        with patch("logging.Logger.debug") as mock_log:
            self._run_one_iteration(c)
        mock_log.assert_called()

    @patch("threading.Thread")
    def test_cleanup_exception_in_items_is_caught(self, mock_thread):
        """Exception in ttl_times.items() is caught by except Exception.
        Replace ttl_times with a MagicMock whose .items() raises."""
        c = AdvancedCache()
        c.cache["a"] = 1
        mock_ttl = MagicMock()
        mock_ttl.items.side_effect = RuntimeError("items failed")
        c.ttl_times = mock_ttl
        c.access_times["a"] = 99999.0
        with patch("logging.Logger.error") as mock_log:
            self._run_one_iteration(c)
        mock_log.assert_called()

    @patch("threading.Thread")
    def test_cleanup_exception_in_remove_is_caught(self, mock_thread):
        """Exception in _remove_key is caught by except Exception."""
        now = time.time()
        c = AdvancedCache()
        c.cache["stale"] = "gone"
        c.ttl_times["stale"] = now - 60
        c.access_times["stale"] = now - 3600
        with patch.object(c, "_remove_key") as mock_rm:
            mock_rm.side_effect = RuntimeError("remove failed")
            with patch("logging.Logger.error") as mock_log:
                self._run_one_iteration(c)
            mock_log.assert_called()


# =========================================================================
# _ScanCache — adaptive TTL subclass (defined in server_setup.py)
# =========================================================================

class TestScanCache:
    @pytest.fixture
    def sc(self):
        c = _ScanCache(max_size=500, default_ttl=1800)
        c.cache.clear()
        c.ttl_times.clear()
        c.access_times.clear()
        return c

    def test_set_default_ttl(self, sc):
        sc.set("k", "v", execution_time=5)
        expiry = sc.ttl_times["k"]
        expected = time.time() + 1800
        assert abs(expiry - expected) < 5

    def test_set_medium_ttl(self, sc):
        sc.set("k", "v", execution_time=30)
        expiry = sc.ttl_times["k"]
        expected = time.time() + 3600
        assert abs(expiry - expected) < 5

    def test_set_long_ttl(self, sc):
        sc.set("k", "v", execution_time=120)
        expiry = sc.ttl_times["k"]
        expected = time.time() + 5400
        assert abs(expiry - expected) < 5

    def test_set_explicit_ttl(self, sc):
        sc.set("k", "v", execution_time=999, ttl=123)
        expiry = sc.ttl_times["k"]
        expected = time.time() + 123
        assert abs(expiry - expected) < 5

    def test_set_boundary_10_seconds(self, sc):
        sc.set("k", "v", execution_time=10)
        expiry = sc.ttl_times["k"]
        expected = time.time() + 1800
        assert abs(expiry - expected) < 5

    def test_set_boundary_60_seconds(self, sc):
        sc.set("k", "v", execution_time=60)
        expiry = sc.ttl_times["k"]
        expected = time.time() + 3600
        assert abs(expiry - expected) < 5

    def test_stats(self, sc):
        sc.set("k", "v", execution_time=5)
        stats = sc.stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert stats["max_size"] == 500

    def test_stats_with_hits(self, sc):
        sc.set("k", "v")
        sc.get("k")
        stats = sc.stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 0

    def test_stats_only_misses(self, sc):
        sc.get("missing")
        stats = sc.stats()
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 1


# =========================================================================
# _cache_key_for — cache key builder
# =========================================================================

class TestCacheKeyFor:
    def test_no_params(self):
        key = _cache_key_for("sid1", "nmap", "10.0.0.1", {})
        assert key == "sid1:nmap:10.0.0.1"

    def test_with_params(self):
        key = _cache_key_for("sid1", "nmap", "10.0.0.1", {"ports": "80,443"})
        assert key.startswith("sid1:nmap:10.0.0.1:")
        assert len(key) > len("sid1:nmap:10.0.0.1:")

    def test_skips_private_and_target(self):
        key = _cache_key_for(
            "sid1", "nmap", "10.0.0.1",
            {"_internal": "x", "target": "10.0.0.1", "ports": "80"},
        )
        assert key.startswith("sid1:nmap:10.0.0.1:")
        assert "_internal" not in key
        assert "target" not in key

    def test_empty_relevant_params(self):
        key = _cache_key_for("sid1", "nmap", "10.0.0.1", {"_private": "val"})
        assert key == "sid1:nmap:10.0.0.1"

    def test_param_order_independence(self):
        k1 = _cache_key_for("s", "nmap", "t", {"b": "2", "a": "1"})
        k2 = _cache_key_for("s", "nmap", "t", {"a": "1", "b": "2"})
        assert k1 == k2


# =========================================================================
# _collect_cached_scans — scan collection from global _scan_cache
# =========================================================================

class TestCollectCachedScans:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        from mcp_core.server_setup import _scan_cache
        _scan_cache.cache.clear()
        _scan_cache.ttl_times.clear()
        _scan_cache.access_times.clear()
        yield

    def test_collects_matching(self):
        from mcp_core.server_setup import _scan_cache
        _scan_cache.cache["sid1:nmap:10.0.0.1"] = {
            "tool": "nmap", "target": "10.0.0.1", "result": {"success": True},
        }
        _scan_cache.cache["sid1:whatweb:10.0.0.1"] = {
            "tool": "whatweb", "target": "10.0.0.1", "result": {"success": True},
        }
        _scan_cache.cache["sid2:nmap:10.0.0.2"] = {
            "tool": "nmap", "target": "10.0.0.2", "result": {"success": True},
        }
        scans = _collect_cached_scans("sid1", "10.0.0.1")
        assert "nmap" in scans
        assert "whatweb" in scans
        assert len(scans) == 2

    def test_no_matching(self):
        scans = _collect_cached_scans("sid99", "no-target")
        assert scans == {}

    def test_skip_missing_tool_field(self):
        from mcp_core.server_setup import _scan_cache
        _scan_cache.cache["sid1:nmap:10.0.0.1"] = {
            "target": "10.0.0.1", "result": {"data": 1},
        }
        scans = _collect_cached_scans("sid1", "10.0.0.1")
        assert scans == {}

    def test_when_result_missing_uses_empty_dict(self):
        from mcp_core.server_setup import _scan_cache
        _scan_cache.cache["sid1:nmap:10.0.0.1"] = {
            "tool": "nmap", "target": "10.0.0.1",
        }
        scans = _collect_cached_scans("sid1", "10.0.0.1")
        assert scans == {"nmap": {}}
