"""
server_core/operational_metrics.py

In-memory operational metrics store for HexStrike AI-PULSE.

Captures per-tool execution data from run_security_tool() telemetry and
exposes aggregated views that satisfy the Phase 4 observability spec:

  - error count by tool
  - timeout count by tool
  - success rate by tool
  - slowest tools (by average duration)
  - cache hit/miss ratio (global)
  - confirmation events (accepted / denied)
  - session state restore count

Design:
  - Pure in-memory, reset on server restart (no persistence required for Phase 4)
  - Thread-safe via RLock (run_security_tool runs in executor threads)
  - Single module-level singleton imported by server_setup.py
  - No external dependencies beyond stdlib
"""

import threading
import time
from typing import Any, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class OperationalMetricsStore:
    """
    Per-tool operational metrics store.

    Intended usage in run_security_tool():

        _op_metrics.record(_telemetry)

    Called once per tool invocation, after the telemetry dict is finalized.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._start_time = time.time()

        # Per-tool counters: {tool_name: {runs, successes, errors, timeouts, cache_hits,
        #                                  session_restores, total_duration, max_duration}}
        self._tools: Dict[str, Dict[str, Any]] = {}

        # Global confirmation counters
        self._confirmations: Dict[str, int] = {
            "accepted": 0,
            "denied":   0,
            "skipped":  0,
        }

        # Global cache counters
        self._cache_hits   = 0
        self._cache_misses = 0

        # Prompt suggestion counter
        self._prompt_suggestions = 0

    # ── Record ────────────────────────────────────────────────────────────────

    def record(self, telemetry: Dict[str, Any]) -> None:
        """
        Record one run_security_tool() telemetry dict.

        Expected keys (all optional — missing keys are treated as falsy/zero):
            tool, success, duration, timed_out, cache_hit, session_state,
            confirmation, prompt_suggested
        """
        tool      = telemetry.get("tool", "unknown")
        success   = bool(telemetry.get("success", False))
        duration  = float(telemetry.get("duration", 0.0))
        timed_out = bool(telemetry.get("timed_out", False))
        cache_hit = bool(telemetry.get("cache_hit", False))
        session   = bool(telemetry.get("session_state", False))
        confirm   = telemetry.get("confirmation")   # None | "accepted" | "denied" | "skipped"
        prompted  = bool(telemetry.get("prompt_suggested", False))

        with self._lock:
            entry = self._tools.setdefault(tool, {
                "runs":             0,
                "successes":        0,
                "errors":           0,
                "timeouts":         0,
                "cache_hits":       0,
                "session_restores": 0,
                "total_duration":   0.0,
                "max_duration":     0.0,
            })

            entry["runs"]           += 1
            entry["total_duration"] += duration
            entry["max_duration"]    = max(entry["max_duration"], duration)

            if success:
                entry["successes"] += 1
            else:
                entry["errors"] += 1

            if timed_out:
                entry["timeouts"] += 1

            if cache_hit:
                entry["cache_hits"] += 1
                self._cache_hits    += 1
            else:
                self._cache_misses  += 1

            if session:
                entry["session_restores"] += 1

            if confirm in self._confirmations:
                self._confirmations[confirm] += 1

            if prompted:
                self._prompt_suggestions += 1

    # ── Views ─────────────────────────────────────────────────────────────────

    def success_rate_by_tool(self) -> List[Dict[str, Any]]:
        """Return tools sorted by success rate ascending (worst first)."""
        with self._lock:
            rows = []
            for tool, e in self._tools.items():
                runs = e["runs"]
                rate = (e["successes"] / runs) if runs else 0.0
                rows.append({
                    "tool":         tool,
                    "runs":         runs,
                    "successes":    e["successes"],
                    "errors":       e["errors"],
                    "success_rate": round(rate, 4),
                })
            rows.sort(key=lambda x: x["success_rate"])
            return rows

    def error_count_by_tool(self) -> List[Dict[str, Any]]:
        """Return tools sorted by error count descending."""
        with self._lock:
            rows = [
                {"tool": t, "errors": e["errors"], "runs": e["runs"]}
                for t, e in self._tools.items()
            ]
            rows.sort(key=lambda x: x["errors"], reverse=True)
            return rows

    def timeout_count_by_tool(self) -> List[Dict[str, Any]]:
        """Return tools sorted by timeout count descending."""
        with self._lock:
            rows = [
                {"tool": t, "timeouts": e["timeouts"], "runs": e["runs"]}
                for t, e in self._tools.items()
                if e["timeouts"] > 0
            ]
            rows.sort(key=lambda x: x["timeouts"], reverse=True)
            return rows

    def slowest_tools(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return top_n tools by average duration descending."""
        with self._lock:
            rows = []
            for tool, e in self._tools.items():
                runs = e["runs"]
                avg  = (e["total_duration"] / runs) if runs else 0.0
                rows.append({
                    "tool":         tool,
                    "avg_duration": round(avg, 3),
                    "max_duration": round(e["max_duration"], 3),
                    "runs":         runs,
                })
            rows.sort(key=lambda x: x["avg_duration"], reverse=True)
            return rows[:top_n]

    def cache_summary(self) -> Dict[str, Any]:
        """Return global cache hit/miss ratio."""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            ratio = (self._cache_hits / total) if total else 0.0
            return {
                "hits":      self._cache_hits,
                "misses":    self._cache_misses,
                "total":     total,
                "hit_ratio": round(ratio, 4),
            }

    def confirmation_summary(self) -> Dict[str, Any]:
        """Return confirmation event counts."""
        with self._lock:
            return dict(self._confirmations)

    @staticmethod
    def system_metrics() -> Dict[str, Any]:
        """Snapshot of system resources (CPU, memory, disk)."""
        if not HAS_PSUTIL:
            return {"status": "unavailable", "reason": "psutil not installed"}
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "cpu_percent": cpu,
                "memory_percent": mem.percent,
                "memory_available_gb": round(mem.available / (1024**3), 1),
                "disk_usage_percent": round(disk.used / disk.total * 100, 1),
                "disk_free_gb": round(disk.free / (1024**3), 1),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)[:100]}

    def summary(self) -> Dict[str, Any]:
        """Full operational summary for the metrics:// resource."""
        with self._lock:
            total_runs = sum(e["runs"] for e in self._tools.values())
            total_ok   = sum(e["successes"] for e in self._tools.values())
            uptime     = round(time.time() - self._start_time, 1)
            return {
                "uptime_seconds":      uptime,
                "total_runs":          total_runs,
                "total_successes":     total_ok,
                "total_errors":        total_runs - total_ok,
                "global_success_rate": round((total_ok / total_runs), 4) if total_runs else 0.0,
                "cache":               self.cache_summary(),
                "confirmations":       self.confirmation_summary(),
                "prompt_suggestions":  self._prompt_suggestions,
                "tools_seen":          sorted(self._tools.keys()),
                "system":              self.system_metrics(),
                "success_rate_by_tool": self.success_rate_by_tool(),
                "error_count_by_tool":  self.error_count_by_tool(),
                "timeout_count_by_tool": self.timeout_count_by_tool(),
                "slowest_tools":        self.slowest_tools(),
            }


# Module-level singleton — imported by server_setup.py
_op_metrics = OperationalMetricsStore()
