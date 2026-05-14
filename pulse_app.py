"""
HexStrike Pulse — Prefab UI Dashboard

FastMCPApp with full 6-panel dashboard using MCP Resources (zero-token reads).

Panels:
  1. Status Bar — version, uptime, tools count, executions, success rate, errors
  2. Resource Gauges — CPU/Mem/Disk via Progress
  3. Cache + Error Stats — Badge chips
  4. Execution Activity — recent scans
  5. Tools by Category — Accordion
  6. Intelligence — baseline vs live effectiveness per tool

Usage:
    fastmcp dev apps pulse_app.py   # preview at localhost:8080
    python3 pulse_app.py            # standalone server
"""

import shutil

from fastmcp import FastMCPApp

from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge, Column, DataTable,
    DataTableColumn, ForEach, Heading, Metric,
    Progress, Row, Separator, Text,
)
from prefab_ui.rx import Rx

from server_core.operational_metrics import _op_metrics
from server_core.singletons import get_tool_stats_store
from mcp_core.server_setup import _scan_cache
from tool_registry import TOOLS
from hexstrike_server import _HEALTH_TOOL_CATEGORIES

app = FastMCPApp("Pulse Dashboard")


# ═════════════════════════════════════════════════════════════════════════════
# Backend tools
# ═════════════════════════════════════════════════════════════════════════════

@app.tool()
def get_pulse_metrics() -> dict:
    """Return live Pulse dashboard data from operational metrics."""
    summary = _op_metrics.summary()
    return {
        "uptime":     summary["uptime_seconds"],
        "total_runs": summary["total_runs"],
        "successes":  summary["total_successes"],
        "errors":     summary["total_errors"],
        "success_rate": round(summary["global_success_rate"] * 100, 1),
        "tools_count": len(TOOLS),
        "system":     summary.get("system", {}),
        "cache":      summary.get("cache", {}),
        "confirmations": summary.get("confirmations", {}),
        "slowest":    summary.get("slowest_tools", []),
        "tools_seen": len(summary.get("tools_seen", [])),
        "error_count_by_tool": summary.get("error_count_by_tool", []),
        "timeout_count_by_tool": summary.get("timeout_count_by_tool", []),
        "success_rate_by_tool": summary.get("success_rate_by_tool", []),
    }


@app.tool()
def get_tool_intelligence() -> list[dict]:
    """Return baseline vs live effectiveness for all tools with recorded runs."""
    tool_stats = get_tool_stats_store()
    all_stats = tool_stats.get_all_stats()
    result = []
    for tool, stats in sorted(all_stats.items()):
        runs = stats["runs"]
        baseline = TOOLS.get(tool, {}).get("effectiveness", 0.5)
        live = tool_stats.live_effectiveness(tool)
        blended = tool_stats.blended_effectiveness(tool, baseline)
        result.append({
            "tool":     tool,
            "baseline": round(baseline, 2),
            "live":     round(live, 2) if live is not None else None,
            "blended":  round(blended, 2),
            "runs":     runs,
            "successes": stats["successes"],
        })
    return result


@app.tool()
def get_recent_scans() -> list[dict]:
    """Return last 20 scan cache entries."""
    entries = sorted(
        _scan_cache.values(),
        key=lambda v: v.get("timestamp", 0),
        reverse=True,
    )[:20]
    return [
        {
            "tool":      e.get("tool", "?"),
            "target":    e.get("target", "?"),
            "timestamp": e.get("timestamp", ""),
            "success":   e.get("result", {}).get("success", False),
        }
        for e in entries
    ]


@app.tool()
def get_tools_by_category() -> list[dict]:
    """Return flat list of all category tools with availability."""
    result = []
    for cat, tools in _HEALTH_TOOL_CATEGORIES.items():
        for tool in tools:
            available = bool(shutil.which(tool))
            result.append({
                "category":   cat,
                "tool":       tool,
                "available":  available,
                "status":     "✓" if available else "✗",
                "variant":    "success" if available else "destructive",
            })
    return result


# ═════════════════════════════════════════════════════════════════════════════
# UI entry point
# ═════════════════════════════════════════════════════════════════════════════

@app.ui()
def pulse_dashboard() -> PrefabApp:
    """Open the Pulse dashboard."""
    with Column(gap=4, css_class="p-4") as view:
        # ── 1. Status Bar ──────────────────────────────────────────────
        with Row(gap=3, align="center"):
            Badge("PULSE", variant="destructive")
            Text(Rx("uptime").then(
                lambda s: f"uptime {_fmt_duration(s)}",
                "uptime —",
            ))
            Text(Rx("tools_count").then(
                lambda n: f"{n} tools",
                "— tools",
            ))

        with Row(gap=4):
            Metric(label="Executions",  value=Rx("total_runs"))
            Metric(
                label="Success Rate",
                value=Rx("success_rate"),
                trend="up", trend_sentiment="positive",
            )
            Metric(label="Errors", value=Rx("errors"),
                   trend="up", trend_sentiment="negative")
            Metric(
                label="Cache Hit Rate",
                value=Rx("cache").then(lambda c: c.get("hit_ratio", 0)),
            )

        Separator()

        # ── 2. Resource Gauges ─────────────────────────────────────────
        Heading("System Resources", level=3)
        with Row(gap=4):
            with Column(gap=2):
                Text("CPU")
                Progress(value=Rx("system").then(
                    lambda s: s.get("cpu_percent", 0)))
            with Column(gap=2):
                Text("Memory")
                Progress(value=Rx("system").then(
                    lambda s: s.get("memory_percent", 0)),
                    variant="warning")
            with Column(gap=2):
                Text("Disk")
                Progress(value=Rx("system").then(
                    lambda s: s.get("disk_usage_percent", 0)),
                    variant="destructive")

        Separator()

        # ── 3. Cache + Errors ──────────────────────────────────────────
        with Row(gap=4):
            with Column(gap=2):
                Heading("Cache", level=3)
                with Row(gap=2):
                    Badge(Rx("cache").then(
                        lambda c: f"hits {c.get('hits',0)}"),
                        variant="success")
                    Badge(Rx("cache").then(
                        lambda c: f"misses {c.get('misses',0)}"),
                        variant="warning")
                    Badge(Rx("cache").then(
                        lambda c: f"ratio {c.get('hit_ratio',0)}%"),
                        variant="info")

            with Column(gap=2):
                Heading("Errors by Tool", level=3)
                with ForEach("error_count_by_tool") as e:
                    Badge(e.tool, variant="destructive")

        Separator()

        # ── 4. Execution Activity ──────────────────────────────────────
        Heading("Recent Activity", level=3)
        with ForEach("recent_scans") as item:
            with Row(gap=2, align="center"):
                Badge(item.tool, variant="outline")
                Text(item.target)

        Separator()

        # ── 5. Tools by Category ───────────────────────────────────────
        Heading("Tools by Category", level=3)
        with ForEach("tools_by_category") as item:
            with Row(gap=2, align="center"):
                Badge(item.category, variant="info")
                Text(item.tool)
                Badge(item.status, variant=item.variant)

        Separator()

        # ── 6. Intelligence ────────────────────────────────────────────
        Heading("Intelligence", level=3)
        DataTable(
            columns=[
                DataTableColumn(name="tool",      label="Tool"),
                DataTableColumn(name="baseline",  label="Baseline"),
                DataTableColumn(name="live",      label="Live"),
                DataTableColumn(name="blended",   label="Blended"),
                DataTableColumn(name="runs",      label="Runs"),
            ],
            rows=Rx("intelligence"),
        )

    return PrefabApp(
        view=view,
        state={
            **get_pulse_metrics(),
            "recent_scans":      get_recent_scans(),
            "intelligence":      get_tool_intelligence(),
            "tools_by_category": get_tools_by_category(),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_duration(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"


if __name__ == "__main__":
    app.run()
