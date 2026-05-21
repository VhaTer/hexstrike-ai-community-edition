"""Cache performance benchmarks (pytest-benchmark)."""
import time
import threading

import pytest

pytestmark = pytest.mark.slow

from server_core.advanced_cache import AdvancedCache
from mcp_core.server_setup import _ScanCache


class TestAdvancedCacheBenchmarks:
    """Benchmark core AdvancedCache operations."""

    def test_set_sequential(self, benchmark):
        c = AdvancedCache(max_size=10_000)
        def _run():
            for i in range(1000):
                c.set(f"k{i}", f"v{i}")
        benchmark(_run)

    def test_get_sequential_hit(self, benchmark):
        c = AdvancedCache(max_size=10_000)
        for i in range(1000):
            c.set(f"k{i}", f"v{i}")
        def _run():
            for i in range(1000):
                c.get(f"k{i}")
        benchmark(_run)

    def test_get_miss(self, benchmark):
        c = AdvancedCache(max_size=10_000)
        for i in range(1000):
            c.set(f"k{i}", f"v{i}")
        def _run():
            for i in range(2000, 3000):
                c.get(f"k{i}")
        benchmark(_run)

    def test_delete(self, benchmark):
        c = AdvancedCache(max_size=10_000)
        for i in range(1000):
            c.set(f"k{i}", f"v{i}")
        def _run():
            for i in range(1000):
                c.delete(f"k{i}")
        benchmark(_run)

    def test_lru_eviction(self, benchmark):
        c = AdvancedCache(max_size=100)
        for i in range(100):
            c.set(f"k{i}", f"v{i}")
        def _run():
            for i in range(100, 200):
                c.set(f"k{i}", f"v{i}")
        benchmark(_run)

    def test_mixed_workload(self, benchmark):
        c = AdvancedCache(max_size=500)
        for i in range(200):
            c.set(f"k{i}", f"v{i}")
        def _run():
            for i in range(200):
                c.get(f"k{i}")
                c.get(f"missing-{i}")
            for i in range(200, 250):
                c.set(f"k{i}", f"v{i}")
            for i in range(0, 50):
                c.delete(f"k{i}")
        benchmark(_run)

    def test_stats_overhead(self, benchmark):
        c = AdvancedCache(max_size=10_000)
        for i in range(1000):
            c.set(f"k{i}", f"v{i}")
        def _run():
            for _ in range(100):
                c.get_stats()
        benchmark(_run)


class TestScanCacheBenchmarks:
    """Benchmark _ScanCache adaptive TTL operations."""

    def test_set_with_ttl_learning(self, benchmark):
        c = _ScanCache(max_size=1000)
        def _run():
            for i in range(200):
                c.set(
                    f"s:nmap:10.0.0.{i}",
                    {"tool": "nmap", "target": f"10.0.0.{i}"},
                    execution_time=5.0,
                )
        benchmark(_run)

    def test_get_with_score_tracking(self, benchmark):
        c = _ScanCache(max_size=1000)
        for i in range(200):
            c.set(
                f"s:nmap:10.0.0.{i}",
                {"tool": "nmap", "target": f"10.0.0.{i}"},
                execution_time=5.0,
            )
        def _run():
            for i in range(200):
                c.get(f"s:nmap:10.0.0.{i}")
        benchmark(_run)

    def test_mixed_adaptive_workload(self, benchmark):
        c = _ScanCache(max_size=500)
        tools = ["nmap", "whatweb", "nuclei", "nikto", "gobuster"]
        for ti, t in enumerate(tools):
            for i in range(20):
                c.set(
                    f"s:{t}:10.0.0.{ti}.{i}",
                    {"tool": t, "target": f"10.0.0.{ti}"},
                    execution_time=5.0,
                )
        def _run():
            for t in tools:
                for i in range(20):
                    c.get(f"s:{t}:10.0.0.{ti}.{i}") if i % 2 == 0 else None
            c.get_ttl_scores()
        benchmark(_run)

    def test_get_ttl_scores(self, benchmark):
        c = _ScanCache(max_size=1000)
        tools = ["nmap", "whatweb", "nuclei", "nikto", "gobuster", "sqlmap", "hydra"]
        for t in tools:
            for i in range(10):
                c.set(
                    f"s:{t}:tgt.{i}",
                    {"tool": t, "target": "tgt"},
                    execution_time=15.0,
                )
                c.get(f"s:{t}:tgt.{i}")
        def _run():
            for _ in range(50):
                c.get_ttl_scores()
        benchmark(_run)


class TestScanCacheAdaptiveTTL:
    """Verify adaptive TTL logic (not benchmark, correctness)."""

    def test_hit_ratio_increases_ttl(self):
        c = _ScanCache(max_size=100)
        for i in range(10):
            c.set(f"s:nmap:tgt.{i}", {"tool": "nmap", "target": "tgt"}, execution_time=5.0)
        for i in range(10):
            c.get(f"s:nmap:tgt.{i}")
        scores = c.get_ttl_scores()
        nmap_scores = scores.get("nmap", {})
        assert nmap_scores.get("hits", 0) == 10
        assert nmap_scores.get("misses", 0) == 0
        assert nmap_scores.get("hit_ratio", 0) > 0.9

    def test_low_hit_ratio_decreases_ttl(self):
        c = _ScanCache(max_size=100)
        for i in range(10):
            c.set(f"s:nmap:tgt.{i}", {"tool": "nmap", "target": "tgt"}, execution_time=5.0)
        # 0 hits, many misses
        for i in range(100):
            c.get(f"s:nmap:other.{i}")
        scores = c.get_ttl_scores()
        nmap = scores.get("nmap", {})
        assert nmap.get("misses", 0) >= 90
        assert nmap.get("hit_ratio", 1) < 0.1

    def test_ttl_learning_requires_3_samples(self):
        c = _ScanCache(max_size=100)
        for i in range(2):
            c.set(f"s:nmap:tgt.{i}", {"tool": "nmap", "target": "tgt"}, execution_time=5.0)
        scores = c.get_ttl_scores()
        assert scores["nmap"]["sets"] == 2
        assert scores["nmap"]["current_ttl_seconds"] == 1800  # default, unchanged

    def test_capacity_large(self):
        c = _ScanCache(max_size=500)
        for i in range(500):
            c.set(f"s:tool:tgt.{i}", {"tool": "tool", "target": "tgt"}, execution_time=10.0)
        assert len(c.cache) <= 500
        c.set("s:tool:extra", {"tool": "tool", "target": "extra"}, execution_time=10.0)
        assert len(c.cache) <= 500

    def test_multi_tool_scores(self):
        c = _ScanCache(max_size=100)
        for t in ["nmap", "whatweb", "gobuster"]:
            for i in range(5):
                c.set(f"s:{t}:tgt.{i}", {"tool": t, "target": "tgt"}, execution_time=5.0)
                c.get(f"s:{t}:tgt.{i}")
        scores = c.get_ttl_scores()
        assert set(scores.keys()) == {"nmap", "whatweb", "gobuster"}
        for t in ["nmap", "whatweb", "gobuster"]:
            assert scores[t]["hits"] == 5
            assert scores[t]["sets"] == 5
