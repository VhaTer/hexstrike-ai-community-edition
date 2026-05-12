import time
from server_core.cache import HexStrikeCache


class TestHexStrikeCache:
    def test_set_and_get(self):
        c = HexStrikeCache()
        c.set("nmap", {"host": "example.com"}, {"port": 80})
        result = c.get("nmap", {"host": "example.com"})
        assert result == {"port": 80}

    def test_miss_returns_none(self):
        c = HexStrikeCache()
        result = c.get("nmap", {"host": "nonexistent"})
        assert result is None

    def test_expired_entry(self):
        c = HexStrikeCache(ttl=0)
        c.set("nmap", {"host": "example.com"}, {"port": 80})
        result = c.get("nmap", {"host": "example.com"})
        assert result is None

    def test_eviction_when_full(self):
        c = HexStrikeCache(max_size=2)
        c.set("a", {}, "x")
        c.set("b", {}, "y")
        c.set("c", {}, "z")
        assert c.get("a", {}) is None
        assert c.get("c", {}) == "z"

    def test_clear_resets_stats(self):
        c = HexStrikeCache()
        c.set("a", {}, "x")
        c.get("a", {})
        c.clear()
        assert len(c.cache) == 0
        assert c.stats["hits"] == 0

    def test_get_stats(self):
        c = HexStrikeCache()
        c.set("a", {}, "x")
        c.get("a", {})
        c.get("b", {})
        stats = c.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
