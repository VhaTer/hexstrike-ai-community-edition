"""Tests for telemetry MCP resources: telemetry://summary, telemetry://recent, telemetry://tools/{tool}.

Validates the TelemetryPipeline methods that back the resource URIs.
"""

import pytest
from server_core.telemetry_pipeline import TelemetryPipeline


@pytest.fixture
def pipe():
    return TelemetryPipeline(jsonl_path=None, max_buffer=100)


class TestTelemetrySummary:
    """telemetry://summary — aggregate telemetry."""

    def test_summary_has_expected_keys(self, pipe):
        s = pipe.summary()
        assert "total_runs" in s
        assert "total_errors" in s
        assert "cache" in s
        assert "confirmations" in s
        assert "global_success_rate" in s

    def test_summary_empty_pipeline(self, pipe):
        s = pipe.summary()
        assert s["total_runs"] == 0
        assert s["total_errors"] == 0

    def test_summary_after_emit(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "nmap", "success": True})
        s = pipe.summary()
        assert s["total_runs"] >= 1

    def test_summary_cache_section_shape(self, pipe):
        s = pipe.summary()
        c = s["cache"]
        assert "hits" in c
        assert "misses" in c

    def test_summary_confirmations_section(self, pipe):
        s = pipe.summary()
        cf = s["confirmations"]
        assert "accepted" in cf
        assert "denied" in cf


class TestTelemetryRecent:
    """telemetry://recent — last N events."""

    def test_recent_events_empty(self, pipe):
        assert pipe.recent_events(10) == []

    def test_recent_events_returns_up_to_n(self, pipe):
        for i in range(5):
            pipe.emit({"event": "tool_execution", "tool": "nmap", "seq": i})
        events = pipe.recent_events(3)
        assert len(events) == 3

    def test_recent_events_ordered_fifo(self, pipe):
        for i in range(3):
            pipe.emit({"event": "tool_execution", "tool": "nmap", "seq": i})
        events = pipe.recent_events(3)
        assert events[0]["seq"] == 0
        assert events[-1]["seq"] == 2

    def test_recent_events_caps_at_buffer_size(self, pipe):
        for i in range(200):
            pipe.emit({"event": "tool_execution", "tool": "nmap", "seq": i})
        events = pipe.recent_events(999)
        assert len(events) <= 100


class TestTelemetryPerTool:
    """telemetry://tools/{tool} — per-tool stats."""

    def test_per_tool_unknown(self, pipe):
        result = pipe.per_tool("nonexistent_tool_xyz")
        assert "error" in result

    def test_per_tool_after_emit(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "nmap", "success": True, "duration": 1.5})
        result = pipe.per_tool("nmap")
        assert "runs" in result
        assert result["runs"] >= 1

    def test_per_tool_includes_duration(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "nmap", "success": True, "duration": 2.0})
        result = pipe.per_tool("nmap")
        assert result["avg_duration"] >= 1.0

    def test_per_tool_multiple_emits_accumulate(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "nmap", "success": True, "duration": 1.0})
        pipe.emit({"event": "tool_execution", "tool": "nmap", "success": False, "duration": 3.0})
        result = pipe.per_tool("nmap")
        assert result["runs"] == 2
        assert result["errors"] == 1

    def test_per_tool_success_rate(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "sqlmap", "success": True, "duration": 0.5})
        pipe.emit({"event": "tool_execution", "tool": "sqlmap", "success": True, "duration": 1.0})
        pipe.emit({"event": "tool_execution", "tool": "sqlmap", "success": False, "duration": 2.0})
        result = pipe.per_tool("sqlmap")
        assert result["success_rate"] == pytest.approx(2 / 3, rel=0.01)
