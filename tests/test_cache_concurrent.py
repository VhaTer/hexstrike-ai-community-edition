"""Concurrent load tests for AdvancedCache and _ScanCache."""
import threading
import time

import pytest

pytestmark = pytest.mark.slow

from server_core.advanced_cache import AdvancedCache
from mcp_core.server_setup import _ScanCache


class TestConcurrentAdvancedCache:
    """Concurrent access patterns on AdvancedCache."""

    def test_concurrent_read_write_10_threads(self):
        c = AdvancedCache(max_size=10_000)
        errors = []
        barrier = threading.Barrier(10)

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(200):
                    key = f"k{ wid * 1000 + i }"
                    c.set(key, f"v{ wid }")
                    val = c.get(key)
                    c.delete(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_read_write_50_threads(self):
        c = AdvancedCache(max_size=5000)
        errors = []
        barrier = threading.Barrier(50)

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(100):
                    key = f"k{ wid * 100 + i }"
                    c.set(key, f"v{ wid }")
                    c.get(key)
                    c.delete(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_get_miss_50_threads(self):
        c = AdvancedCache(max_size=1000)
        errors = []
        barrier = threading.Barrier(50)

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(200):
                    c.get(f"nonexistent-{wid}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"
        assert c.miss_count == 50 * 200

    def test_concurrent_read_only_50_threads(self):
        c = AdvancedCache(max_size=5000)
        for i in range(2000):
            c.set(f"k{i}", f"v{i}")
        errors = []
        barrier = threading.Barrier(50)

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(100):
                    c.get(f"k{(i * 20 + wid) % 2000}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_eviction_under_load(self):
        c = AdvancedCache(max_size=50)  # tiny — triggers eviction
        barrier = threading.Barrier(10)
        errors = []

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(200):
                    c.set(f"k{ wid * 200 + i }", f"v{ wid }")
                    c.get(f"k{ wid * 200 + i }")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(c.cache) <= 50

    def test_concurrent_mixed_operations(self):
        c = AdvancedCache(max_size=1000)
        barrier = threading.Barrier(20)
        errors = []

        def writer(wid: int):
            barrier.wait()
            try:
                for i in range(100):
                    c.set(f"w{ wid }-{i}", f"v{ wid }")
            except Exception as e:
                errors.append(e)

        def reader(wid: int):
            barrier.wait()
            try:
                for i in range(100):
                    c.get(f"w{ wid }-{i}")
            except Exception as e:
                errors.append(e)

        def deleter(wid: int):
            barrier.wait()
            try:
                for i in range(100):
                    c.delete(f"w{ wid }-{i}")
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(10)]
            + [threading.Thread(target=reader, args=(i,)) for i in range(5)]
            + [threading.Thread(target=deleter, args=(i,)) for i in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"


class TestConcurrentScanCache:
    """Concurrent access patterns on _ScanCache with adaptive TTL."""

    def test_concurrent_ttl_learning(self):
        c = _ScanCache(max_size=500)
        barrier = threading.Barrier(10)
        errors = []

        def worker(wid: int):
            barrier.wait()
            try:
                tool = ["nmap", "whatweb", "nuclei", "nikto", "gobuster"][wid % 5]
                for i in range(100):
                    key = f"s:{tool}:target-{wid}-{i}"
                    c.set(key, {"tool": tool, "target": "target"}, execution_time=15.0)
                    c.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        scores = c.get_ttl_scores()
        assert len(scores) >= 5  # at least the 5 tools used

    def test_concurrent_ttl_scores_race(self):
        c = _ScanCache(max_size=200)
        barrier = threading.Barrier(10)
        errors = []

        def writer(wid: int):
            barrier.wait()
            try:
                for i in range(50):
                    c.set(f"s:tool:tgt-{wid}-{i}", {"tool": "tool", "target": "tgt"}, execution_time=5.0)
            except Exception as e:
                errors.append(e)

        def reader(wid: int):
            barrier.wait()
            try:
                for i in range(50):
                    c.get(f"s:tool:tgt-{wid}-{i}")
            except Exception as e:
                errors.append(e)

        def scorer(wid: int):
            barrier.wait()
            try:
                for _ in range(20):
                    c.get_ttl_scores()
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(4)]
            + [threading.Thread(target=reader, args=(i,)) for i in range(4)]
            + [threading.Thread(target=scorer, args=(i,)) for i in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_high_tool_variety(self):
        c = _ScanCache(max_size=1000)
        tools = [f"tool-{i}" for i in range(50)]
        barrier = threading.Barrier(20)
        errors = []

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(20):
                    tool = tools[(wid * 20 + i) % 50]
                    key = f"s:{tool}:tgt-{wid}-{i}"
                    c.set(key, {"tool": tool, "target": "tgt"}, execution_time=30.0)
                    c.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        scores = c.get_ttl_scores()
        assert len(scores) <= 50
