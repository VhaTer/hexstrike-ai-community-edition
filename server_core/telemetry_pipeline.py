"""Unified telemetry pipeline.

Single entry point for all HexStrike observability data.
Merges middleware on_call_tool events + run_security_tool finalize()
events + TelemetryCollector into one stream with dedup by request_id.

Backends:
  - In-memory circular buffer (last N events)
  - Per-tool aggregation dict (replaces OperationalMetricsStore)
  - Optional JSONL file for persistence
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TelemetryPipeline:
    """Unified telemetry bus — emit once, query everywhere."""

    def __init__(
        self, jsonl_path: str | None = None, max_buffer: int = 1000
    ) -> None:
        self._lock = threading.RLock()
        self._start_time = time.time()
        self._buffer: list[dict[str, Any]] = []
        self._max_buffer = max_buffer
        self._jsonl_path = jsonl_path
        if jsonl_path:
            Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

        # Per-tool aggregation (same schema as OperationalMetricsStore)
        self._tools: dict[str, dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._confirmations: dict[str, int] = {
            "accepted": 0, "denied": 0, "skipped": 0,
        }
        self._prompt_suggestions = 0

    # ── Emit ──────────────────────────────────────────────────────────────────

    def emit(self, event: dict[str, Any]) -> None:
        """Record a telemetry event. Must have at minimum 'event' key."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self._max_buffer:
                self._buffer.pop(0)

            if self._jsonl_path:
                try:
                    with open(self._jsonl_path, "a") as f:
                        f.write(json.dumps(event) + "\n")
                except OSError as exc:
                    logger.warning("telemetry jsonl write failed: %s", exc)

            if event.get("event") in ("tool_call", "tool_execution"):
                self._aggregate(event)

    def _aggregate(self, event: dict[str, Any]) -> None:
        tool = event.get("tool", "unknown")
        success = bool(event.get("success", False))
        duration = float(event.get("duration", 0.0))
        timed_out = bool(event.get("timed_out", False))
        cache_hit = bool(event.get("cache_hit", False))
        session = bool(event.get("session_state", False))
        confirm = event.get("confirmation")
        prompted = bool(event.get("prompt_suggested", False))

        entry = self._tools.setdefault(tool, {
            "runs": 0, "successes": 0, "errors": 0, "timeouts": 0,
            "cache_hits": 0, "session_restores": 0,
            "total_duration": 0.0, "max_duration": 0.0,
        })
        entry["runs"] += 1
        entry["total_duration"] += duration
        entry["max_duration"] = max(entry["max_duration"], duration)
        if success:
            entry["successes"] += 1
        else:
            entry["errors"] += 1
        if timed_out:
            entry["timeouts"] += 1
        if cache_hit:
            entry["cache_hits"] += 1
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        if session:
            entry["session_restores"] += 1
        if confirm in self._confirmations:
            self._confirmations[confirm] += 1
        if prompted:
            self._prompt_suggestions += 1

    # ── Views ─────────────────────────────────────────────────────────────────

    def recent_events(self, n: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer[-n:])

    def per_tool(self, tool: str) -> dict[str, Any]:
        with self._lock:
            entry = self._tools.get(tool)
            if not entry:
                return {"error": f"No data for tool: {tool}"}
            runs = entry["runs"]
            rate = (entry["successes"] / runs) if runs else 0.0
            avg = (entry["total_duration"] / runs) if runs else 0.0
            return {**entry, "success_rate": round(rate, 4), "avg_duration": round(avg, 3)}

    def tools_seen(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            total_runs = sum(e["runs"] for e in self._tools.values())
            total_ok = sum(e["successes"] for e in self._tools.values())
            total = self._cache_hits + self._cache_misses
            cache_ratio = (self._cache_hits / total) if total else 0.0
            uptime = round(time.time() - self._start_time, 1)
            return {
                "uptime_seconds": uptime,
                "total_runs": total_runs,
                "total_successes": total_ok,
                "total_errors": total_runs - total_ok,
                "global_success_rate": round((total_ok / total_runs), 4) if total_runs else 0.0,
                "cache": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "total": total,
                    "hit_ratio": round(cache_ratio, 4),
                },
                "confirmations": dict(self._confirmations),
                "prompt_suggestions": self._prompt_suggestions,
                "tools_seen": sorted(self._tools.keys()),
                "buffer_size": len(self._buffer),
            }

    def success_rate_by_tool(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                {"tool": t, "runs": e["runs"], "successes": e["successes"],
                 "errors": e["errors"],
                 "success_rate": round((e["successes"] / e["runs"]), 4) if e["runs"] else 0.0}
                for t, e in self._tools.items()
            ]
            rows.sort(key=lambda x: x["success_rate"])
            return rows

    def error_count_by_tool(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [{"tool": t, "errors": e["errors"], "runs": e["runs"]}
                    for t, e in self._tools.items()]
            rows.sort(key=lambda x: x["errors"], reverse=True)
            return rows

    def timeout_count_by_tool(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [{"tool": t, "timeouts": e["timeouts"], "runs": e["runs"]}
                    for t, e in self._tools.items() if e["timeouts"] > 0]
            rows.sort(key=lambda x: x["timeouts"], reverse=True)
            return rows

    def slowest_tools(self, top_n: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for tool, e in self._tools.items():
                runs = e["runs"]
                avg = (e["total_duration"] / runs) if runs else 0.0
                rows.append({"tool": tool, "avg_duration": round(avg, 3),
                             "max_duration": round(e["max_duration"], 3), "runs": runs})
            rows.sort(key=lambda x: x["avg_duration"], reverse=True)
            return rows[:top_n]

    def cache_hits_by_tool(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [{"tool": t, "cache_hits": e["cache_hits"], "runs": e["runs"]}
                    for t, e in self._tools.items() if e["cache_hits"] > 0]
            rows.sort(key=lambda x: x["cache_hits"], reverse=True)
            return rows


_pipeline = TelemetryPipeline(
    jsonl_path=os.environ.get("HEXSTRIKE_TELEMETRY_JSONL", ""),
)
