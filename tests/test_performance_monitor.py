"""Coverage for performance_monitor.py — 67% → 100%."""

from unittest.mock import patch

import pytest

from server_core.performance_monitor import PerformanceMonitor


@pytest.fixture
def pm():
    return PerformanceMonitor()


class TestMonitorSystemResources:
    def test_returns_resource_dict(self, pm):
        with patch("server_core.performance_monitor.psutil.cpu_percent",
                   return_value=45.0), \
             patch("server_core.performance_monitor.psutil.virtual_memory") as vm, \
             patch("server_core.performance_monitor.psutil.disk_usage") as du, \
             patch("server_core.performance_monitor.psutil.net_io_counters") as net:
            vm.return_value.percent = 60.0
            du.return_value.percent = 70.0
            net.return_value.bytes_sent = 1000
            net.return_value.bytes_recv = 2000
            r = pm.monitor_system_resources()
        assert r["cpu_percent"] == 45.0
        assert r["memory_percent"] == 60.0
        assert r["disk_percent"] == 70.0
        assert r["network_bytes_sent"] == 1000
        assert r["network_bytes_recv"] == 2000
        assert "timestamp" in r

    def test_psutil_error_returns_empty(self, pm):
        with patch("server_core.performance_monitor.psutil.cpu_percent",
                   side_effect=Exception("no perm")):
            r = pm.monitor_system_resources()
        assert r == {}


class TestOptimizeBasedOnResources:
    def test_no_optimization_when_usage_low(self, pm):
        r = pm.optimize_based_on_resources(
            {"threads": 4, "delay": 1.0},
            {"cpu_percent": 10, "memory_percent": 20}
        )
        assert r["threads"] == 4
        assert r["_optimizations_applied"] == []

    def test_high_cpu_reduces_threads(self, pm):
        r = pm.optimize_based_on_resources(
            {"threads": 8, "delay": 1.0},
            {"cpu_percent": 90}
        )
        assert r["threads"] == 4
        assert r["delay"] == 2.0
        assert "Reduced threads" in r["_optimizations_applied"][0]

    def test_high_cpu_no_threads_key(self, pm):
        r = pm.optimize_based_on_resources(
            {"delay": 1.0},
            {"cpu_percent": 90}
        )
        assert r["delay"] == 2.0
        assert len(r["_optimizations_applied"]) == 1

    def test_high_memory_reduces_batch_size(self, pm):
        r = pm.optimize_based_on_resources(
            {"batch_size": 100},
            {"memory_percent": 90}
        )
        assert r["batch_size"] == 60
        assert "Reduced batch size" in r["_optimizations_applied"][0]

    def test_high_memory_no_batch_key(self, pm):
        r = pm.optimize_based_on_resources(
            {},
            {"memory_percent": 90}
        )
        assert r["_optimizations_applied"] == []

    def test_high_network_reduces_connections(self, pm):
        r = pm.optimize_based_on_resources(
            {"concurrent_connections": 10},
            {"network_bytes_sent": 2000000}
        )
        assert r["concurrent_connections"] == 7
        assert "Reduced concurrent connections" in r["_optimizations_applied"][0]

    def test_high_network_no_connections_key(self, pm):
        r = pm.optimize_based_on_resources(
            {},
            {"network_bytes_sent": 2000000}
        )
        assert r["_optimizations_applied"] == []

    def test_low_network_no_optimization(self, pm):
        r = pm.optimize_based_on_resources(
            {"concurrent_connections": 10},
            {"network_bytes_sent": 500}
        )
        assert r["concurrent_connections"] == 10

    def test_combined_optimizations(self, pm):
        r = pm.optimize_based_on_resources(
            {"threads": 4, "concurrent_connections": 5},
            {"cpu_percent": 90, "network_bytes_sent": 3000000}
        )
        assert any("Reduced threads" in x for x in r["_optimizations_applied"])
        assert any("Reduced concurrent connections" in x for x in r["_optimizations_applied"])

    def test_threads_min_one(self, pm):
        r = pm.optimize_based_on_resources(
            {"threads": 1, "delay": 1.0},
            {"cpu_percent": 90}
        )
        assert r["threads"] == 1

    def test_batch_size_min_one(self, pm):
        r = pm.optimize_based_on_resources(
            {"batch_size": 1},
            {"memory_percent": 90}
        )
        assert r["batch_size"] == 1

    def test_concurrent_connections_min_one(self, pm):
        r = pm.optimize_based_on_resources(
            {"concurrent_connections": 1},
            {"cpu_percent": 10, "network_bytes_sent": 2000000}
        )
        assert r["concurrent_connections"] == 1
