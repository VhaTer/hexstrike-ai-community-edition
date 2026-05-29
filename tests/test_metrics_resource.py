"""Tests for metrics MCP resource: metrics://tools.

Validates that _op_metrics.summary() returns the expected structure.
"""

import pytest
from server_core.operational_metrics import OperationalMetricsStore


@pytest.fixture
def store():
    return OperationalMetricsStore()


class TestMetricsResource:
    """metrics://tools — operational metrics shape and content."""

    def test_summary_has_expected_top_keys(self, store):
        s = store.summary()
        assert "total_runs" in s
        assert "total_errors" in s
        assert "global_success_rate" in s
        assert "cache" in s
        assert "system" in s

    def test_summary_empty_returns_zero_counts(self, store):
        s = store.summary()
        assert s["total_runs"] == 0
        assert s["total_errors"] == 0

    def test_summary_after_record(self, store):
        store.record({"tool": "nmap", "success": True, "duration": 1.5, "target": "x"})
        s = store.summary()
        assert s["total_runs"] >= 1

    def test_summary_success_rate_empty(self, store):
        s = store.summary()
        assert s["global_success_rate"] == 0.0

    def test_summary_cache_summary_has_hits_misses(self, store):
        s = store.summary()
        c = s["cache"]
        assert "hits" in c
        assert "misses" in c
        assert "total" in c

    def test_summary_system_has_cpu_memory_disk(self, store):
        s = store.summary()
        sys = s["system"]
        assert "cpu_percent" in sys
        assert "memory_percent" in sys
        assert "disk_usage_percent" in sys

    def test_error_count_by_tool_empty_when_no_errors(self, store):
        assert store.error_count_by_tool() == []

    def test_timeout_count_by_tool_empty_when_no_timeouts(self, store):
        assert store.timeout_count_by_tool() == []

    def test_success_rate_by_tool_empty_when_no_runs(self, store):
        assert store.success_rate_by_tool() == []

    def test_slowest_tools_empty_when_no_runs(self, store):
        assert store.slowest_tools() == []

    def test_cache_hits_by_tool_empty_when_no_hits(self, store):
        assert store.cache_hits_by_tool() == []
