"""
tests/test_telemetry_pipeline.py

Unit tests for server_core/telemetry_pipeline.py (TelemetryPipeline).

Covers:
  - emit() stores events in buffer
  - emit() aggregates tool_execution events into per-tool counters
  - summary() returns all top-level keys
  - per_tool() returns stats for a specific tool
  - per_tool() returns error for unknown tool
  - success_rate_by_tool() sorts worst-first
  - error_count_by_tool() sorts highest-first
  - timeout_count_by_tool() excludes zero-timeout tools
  - slowest_tools() sorts by avg duration, honours top_n
  - cache_hits_by_tool() only shows tools with hits
  - recent_events() returns last N events
  - Thread safety: concurrent emits don't corrupt counters
  - Edge cases: empty pipeline, single event, all-success, all-failure
"""

import threading
import pytest
from server_core.telemetry_pipeline import TelemetryPipeline


@pytest.fixture
def pipe():
    return TelemetryPipeline(jsonl_path=None, max_buffer=100)


class TestTelemetryPipeline:
    """TelemetryPipeline unit tests."""

    def test_empty_summary(self, pipe):
        s = pipe.summary()
        assert s["total_runs"] == 0
        assert s["total_errors"] == 0
        assert s["tools_seen"] == []

    def test_single_tool_execution(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "nmap",
            "success": True,
            "duration": 1.5,
            "timed_out": False,
            "cache_hit": False,
            "session_state": False,
            "confirmation": None,
            "prompt_suggested": False,
        })
        s = pipe.summary()
        assert s["total_runs"] == 1
        assert s["total_successes"] == 1
        assert s["tools_seen"] == ["nmap"]

    def test_per_tool_returns_stats(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "nmap",
            "success": True,
            "duration": 2.0,
        })
        stats = pipe.per_tool("nmap")
        assert stats["runs"] == 1
        assert stats["successes"] == 1
        assert stats["success_rate"] == 1.0

    def test_per_tool_unknown(self, pipe):
        stats = pipe.per_tool("nonexistent")
        assert "error" in stats

    def test_failed_execution_counts_as_error(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "hydra",
            "success": False,
            "duration": 5.0,
        })
        stats = pipe.per_tool("hydra")
        assert stats["runs"] == 1
        assert stats["successes"] == 0
        assert stats["errors"] == 1

    def test_timeout_tracking(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "nmap",
            "success": False,
            "duration": 30.0,
            "timed_out": True,
        })
        stats = pipe.per_tool("nmap")
        assert stats["timeouts"] == 1

    def test_cache_hit_tracking(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "nmap",
            "success": True,
            "duration": 0.1,
            "cache_hit": True,
        })
        s = pipe.summary()
        assert s["cache"]["hits"] == 1
        assert s["cache"]["misses"] == 0

    def test_cache_miss_tracking(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "nmap",
            "success": True,
            "duration": 1.0,
            "cache_hit": False,
        })
        s = pipe.summary()
        assert s["cache"]["hits"] == 0
        assert s["cache"]["misses"] == 1

    def test_confirmation_tracking(self, pipe):
        pipe.emit({
            "event": "tool_execution",
            "tool": "metasploit",
            "success": False,
            "duration": 0.5,
            "confirmation": "denied",
        })
        s = pipe.summary()
        assert s["confirmations"]["denied"] == 1

    def test_success_rate_by_tool_order(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "a", "success": True, "duration": 0.1})
        pipe.emit({"event": "tool_execution", "tool": "b", "success": False, "duration": 0.1})
        pipe.emit({"event": "tool_execution", "tool": "c", "success": True, "duration": 0.1})
        rows = pipe.success_rate_by_tool()
        # Worst first
        assert rows[0]["tool"] == "b"
        assert rows[0]["success_rate"] == 0.0

    def test_error_count_by_tool_order(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "a", "success": True, "duration": 0.1})
        pipe.emit({"event": "tool_execution", "tool": "b", "success": False, "duration": 0.1})
        pipe.emit({"event": "tool_execution", "tool": "b", "success": False, "duration": 0.1})
        rows = pipe.error_count_by_tool()
        assert rows[0]["tool"] == "b"

    def test_timeout_count_excludes_zero(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "a", "success": True, "duration": 0.1, "timed_out": False})
        pipe.emit({"event": "tool_execution", "tool": "b", "success": False, "duration": 5.0, "timed_out": True})
        rows = pipe.timeout_count_by_tool()
        assert len(rows) == 1
        assert rows[0]["tool"] == "b"

    def test_slowest_tools_order(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "fast", "success": True, "duration": 0.5})
        pipe.emit({"event": "tool_execution", "tool": "slow", "success": True, "duration": 10.0})
        rows = pipe.slowest_tools(top_n=5)
        assert rows[0]["tool"] == "slow"

    def test_cache_hits_by_tool(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "a", "success": True, "duration": 0.1, "cache_hit": True})
        pipe.emit({"event": "tool_execution", "tool": "b", "success": True, "duration": 0.1, "cache_hit": False})
        rows = pipe.cache_hits_by_tool()
        assert len(rows) == 1
        assert rows[0]["tool"] == "a"

    def test_recent_events(self, pipe):
        pipe.emit({"event": "tool_execution", "tool": "a", "duration": 0.1})
        pipe.emit({"event": "tool_execution", "tool": "b", "duration": 0.2})
        recent = pipe.recent_events(5)
        assert len(recent) == 2
        assert recent[-1]["tool"] == "b"

    def test_buffer_trimming(self, pipe):
        for i in range(200):
            pipe.emit({"event": "tool_execution", "tool": f"t{i}", "duration": 0.1})
        assert len(pipe._buffer) == pipe._max_buffer  # 100

    def test_tool_call_event_is_aggregated(self, pipe):
        """Middleware tool_call events are aggregated for protocol-level stats."""
        pipe.emit({"event": "tool_call", "tool": "nmap", "duration_ms": 500})
        s = pipe.summary()
        assert s["total_runs"] == 1  # aggregated same as tool_execution

    def test_concurrent_emits(self, pipe):
        """Thread safety: concurrent emits don't corrupt counters."""
        errors = []

        def emit_worker():
            try:
                for _ in range(100):
                    pipe.emit({
                        "event": "tool_execution",
                        "tool": "nmap",
                        "success": True,
                        "duration": 0.1,
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety error: {errors}"
        s = pipe.summary()
        assert s["total_runs"] == 400
