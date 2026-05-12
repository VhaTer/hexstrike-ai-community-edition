"""
tests/test_operational_metrics.py

Unit tests for server_core/operational_metrics.py

Covers:
  - record() accumulates per-tool counters correctly
  - success_rate_by_tool() sorts worst-first
  - error_count_by_tool() sorts highest-first
  - timeout_count_by_tool() excludes zero-timeout tools
  - slowest_tools() sorts by avg duration, honours top_n
  - cache_summary() computes hit ratio
  - confirmation_summary() counts accepted/denied/skipped
  - summary() aggregates everything
  - Thread safety: concurrent records don't corrupt counters
  - Edge cases: empty store, single run, all-success, all-failure
"""

import threading
import pytest
from unittest.mock import patch
from server_core.operational_metrics import OperationalMetricsStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tel(
    tool="nmap",
    success=True,
    duration=1.0,
    timed_out=False,
    cache_hit=False,
    session_state=False,
    confirmation="skipped",
    prompt_suggested=False,
):
    return {
        "tool":             tool,
        "success":          success,
        "duration":         duration,
        "timed_out":        timed_out,
        "cache_hit":        cache_hit,
        "session_state":    session_state,
        "confirmation":     confirmation,
        "prompt_suggested": prompt_suggested,
    }


# ---------------------------------------------------------------------------
# record() — basic accumulation
# ---------------------------------------------------------------------------

class TestRecord:

    def test_single_success(self):
        m = OperationalMetricsStore()
        m.record(_tel("nmap", success=True, duration=2.0))
        entry = m._tools["nmap"]
        assert entry["runs"] == 1
        assert entry["successes"] == 1
        assert entry["errors"] == 0
        assert entry["total_duration"] == 2.0
        assert entry["max_duration"] == 2.0

    def test_single_failure(self):
        m = OperationalMetricsStore()
        m.record(_tel("sqlmap", success=False, duration=0.5))
        entry = m._tools["sqlmap"]
        assert entry["runs"] == 1
        assert entry["successes"] == 0
        assert entry["errors"] == 1

    def test_multiple_runs_accumulate(self):
        m = OperationalMetricsStore()
        for i in range(5):
            m.record(_tel("nikto", success=(i % 2 == 0), duration=float(i + 1)))
        entry = m._tools["nikto"]
        assert entry["runs"] == 5
        assert entry["successes"] == 3   # i=0,2,4
        assert entry["errors"] == 2      # i=1,3
        assert entry["total_duration"] == 15.0  # 1+2+3+4+5
        assert entry["max_duration"] == 5.0

    def test_timeout_counted(self):
        m = OperationalMetricsStore()
        m.record(_tel("masscan", success=False, timed_out=True))
        assert m._tools["masscan"]["timeouts"] == 1

    def test_no_timeout_not_counted(self):
        m = OperationalMetricsStore()
        m.record(_tel("masscan", success=False, timed_out=False))
        assert m._tools["masscan"]["timeouts"] == 0

    def test_cache_hit_increments_global_and_tool(self):
        m = OperationalMetricsStore()
        m.record(_tel("nmap", cache_hit=True))
        assert m._cache_hits == 1
        assert m._cache_misses == 0
        assert m._tools["nmap"]["cache_hits"] == 1

    def test_cache_miss_increments_global_miss(self):
        m = OperationalMetricsStore()
        m.record(_tel("nmap", cache_hit=False))
        assert m._cache_hits == 0
        assert m._cache_misses == 1

    def test_session_restore_counted(self):
        m = OperationalMetricsStore()
        m.record(_tel("nmap", session_state=True))
        assert m._tools["nmap"]["session_restores"] == 1

    def test_confirmation_accepted(self):
        m = OperationalMetricsStore()
        m.record(_tel("aireplay_ng", confirmation="accepted"))
        assert m._confirmations["accepted"] == 1
        assert m._confirmations["denied"] == 0

    def test_confirmation_denied(self):
        m = OperationalMetricsStore()
        m.record(_tel("metasploit", confirmation="denied"))
        assert m._confirmations["denied"] == 1

    def test_confirmation_none_ignored(self):
        m = OperationalMetricsStore()
        m.record(_tel("nmap", confirmation=None))
        assert sum(m._confirmations.values()) == 0

    def test_prompt_suggested_counted(self):
        m = OperationalMetricsStore()
        m.record(_tel("httpx", prompt_suggested=True))
        assert m._prompt_suggestions == 1

    def test_missing_keys_default_gracefully(self):
        """record() must not raise when telemetry dict is sparse."""
        m = OperationalMetricsStore()
        m.record({"tool": "sparse"})
        assert m._tools["sparse"]["runs"] == 1
        assert m._tools["sparse"]["successes"] == 0

    def test_unknown_tool_creates_entry(self):
        m = OperationalMetricsStore()
        m.record(_tel("brand_new_tool"))
        assert "brand_new_tool" in m._tools


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

class TestViews:

    def _populated(self):
        m = OperationalMetricsStore()
        # nmap: 4 runs, 3 success → 75%
        for i in range(4):
            m.record(_tel("nmap", success=(i != 3), duration=2.0))
        # sqlmap: 2 runs, 0 success → 0%
        m.record(_tel("sqlmap", success=False, duration=5.0))
        m.record(_tel("sqlmap", success=False, duration=7.0, timed_out=True))
        # nikto: 1 run, 1 success → 100%
        m.record(_tel("nikto", success=True, duration=0.5))
        return m

    def test_success_rate_sorted_worst_first(self):
        m = self._populated()
        rates = m.success_rate_by_tool()
        rate_values = [r["success_rate"] for r in rates]
        assert rate_values == sorted(rate_values)   # ascending = worst first
        assert rates[0]["tool"] == "sqlmap"         # 0% worst
        assert rates[-1]["tool"] == "nikto"         # 100% best

    def test_error_count_sorted_highest_first(self):
        m = self._populated()
        errors = m.error_count_by_tool()
        assert errors[0]["tool"] in ("sqlmap", "nmap")  # sqlmap=2 errors, nmap=1
        assert errors[0]["errors"] >= errors[1]["errors"]

    def test_timeout_count_excludes_zero(self):
        m = self._populated()
        timeouts = m.timeout_count_by_tool()
        tools = [t["tool"] for t in timeouts]
        assert "sqlmap" in tools   # 1 timeout
        assert "nmap" not in tools  # 0 timeouts
        assert "nikto" not in tools

    def test_slowest_tools_sorted_desc(self):
        m = self._populated()
        slow = m.slowest_tools()
        avgs = [s["avg_duration"] for s in slow]
        assert avgs == sorted(avgs, reverse=True)
        # sqlmap avg = (5+7)/2 = 6.0, nmap avg = 2.0, nikto = 0.5
        assert slow[0]["tool"] == "sqlmap"

    def test_slowest_tools_honours_top_n(self):
        m = self._populated()
        slow = m.slowest_tools(top_n=1)
        assert len(slow) == 1

    def test_cache_summary_ratio(self):
        m = OperationalMetricsStore()
        m.record(_tel("nmap", cache_hit=True))
        m.record(_tel("nmap", cache_hit=True))
        m.record(_tel("nmap", cache_hit=False))
        cs = m.cache_summary()
        assert cs["hits"] == 2
        assert cs["misses"] == 1
        assert cs["total"] == 3
        assert abs(cs["hit_ratio"] - 2/3) < 0.001

    def test_cache_summary_empty(self):
        m = OperationalMetricsStore()
        cs = m.cache_summary()
        assert cs["hit_ratio"] == 0.0
        assert cs["total"] == 0

    def test_confirmation_summary(self):
        m = OperationalMetricsStore()
        m.record(_tel("aireplay_ng", confirmation="accepted"))
        m.record(_tel("metasploit", confirmation="denied"))
        m.record(_tel("nmap", confirmation="skipped"))
        cs = m.confirmation_summary()
        assert cs["accepted"] == 1
        assert cs["denied"] == 1
        assert cs["skipped"] == 1

    def test_summary_keys_complete(self):
        REQUIRED = {
            "uptime_seconds", "total_runs", "total_successes", "total_errors",
            "global_success_rate", "cache", "confirmations", "prompt_suggestions",
            "tools_seen", "success_rate_by_tool", "error_count_by_tool",
            "timeout_count_by_tool", "slowest_tools",
        }
        m = OperationalMetricsStore()
        m.record(_tel("nmap"))
        s = m.summary()
        missing = REQUIRED - set(s.keys())
        assert not missing, f"Missing summary keys: {missing}"

    def test_summary_empty_store(self):
        m = OperationalMetricsStore()
        s = m.summary()
        assert s["total_runs"] == 0
        assert s["global_success_rate"] == 0.0
        assert s["tools_seen"] == []


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_records_no_corruption(self):
        """100 threads each record 10 runs — total must be exactly 1000."""
        m = OperationalMetricsStore()
        barrier = threading.Barrier(100)

        def worker():
            barrier.wait()
            for _ in range(10):
                m.record(_tel("nmap", success=True, cache_hit=False))

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert m._tools["nmap"]["runs"] == 1000
        assert m._tools["nmap"]["successes"] == 1000
        assert m._cache_misses == 1000


class TestSystemMetrics:
    def test_system_metrics_exception(self):
        with patch("server_core.operational_metrics.psutil") as mock_psutil:
            mock_psutil.cpu_percent.side_effect = RuntimeError("oops")
            result = OperationalMetricsStore.system_metrics()
            assert result["status"] == "error"

    def test_system_metrics_unavailable_without_psutil(self):
        import importlib
        with patch.dict('sys.modules', {'psutil': None}):
            mod = importlib.import_module('server_core.operational_metrics')
            importlib.reload(mod)
            mod2 = importlib.import_module('server_core.operational_metrics')
            importlib.reload(mod2)
            result = mod2.OperationalMetricsStore.system_metrics()
            assert result == {"status": "unavailable", "reason": "psutil not installed"}
