"""
HexStrike Pulse — Prefab UI Dashboard

FastMCPApp with OPS-focused dashboard panels.

Panels (Session B — Header + Scope + Surface + Findings):
  1. Header — version, uptime, RAM, tools, server status
  2. Scope — active target, tools used on it, last scan time
  3. Surface — open ports, services, technologies, risk level
  4. Findings — vulnerabilities from nuclei/nikto
  5. Recent Activity ( -> Historique in Session C)
  6. Intelligence DataTable (kept)

Usage:
    # Via Claude Desktop (stdio):
    python3 hexstrike_mcp.py

    # Via FastMCP dev server (validation only):
    fastmcp dev apps pulse_app.py

    # Via Prefab serve (preview in browser):
    prefab serve debug_app.py
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from fastmcp import FastMCPApp

from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Accordion, AccordionItem,
    Alert, AlertDescription, AlertTitle,
    Badge, Card, CardContent, Code,
    Column, DataTable, DataTableColumn, Div, ForEach,
    Grid, Heading, Icon, Link, Markdown, Metric, Muted,
    Progress, Row, Separator, Tab, Table, TableBody,
    TableCaption, TableCell, TableFooter, TableHead,
    TableHeader, TableRow, Tabs, Text, Tooltip,
)
from prefab_ui.components.charts import BarChart, ChartSeries, Sparkline
from prefab_ui.rx import Rx

from config import _config as app_config
from mcp_core.server_setup import _scan_cache, _rate_limit_events, _optimizer, TOOL_TIMEOUTS
from server_core.operational_metrics import _op_metrics
from server_core.singletons import enhanced_process_manager, error_handler, get_decision_engine, get_target_store, get_tool_stats_store, get_ctf_manager
from server_core.exploit_rules import suggest_exploit, ESTIMATED_TIMES, compute_layer2_score
from tool_registry import TOOLS
from mcp_core.tool_registry_v2 import _registry

logger = logging.getLogger(__name__)

app = FastMCPApp("Pulse Dashboard")

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
_TECH_KEYWORDS = {
    "wordpress": "WordPress", "joomla": "Joomla", "drupal": "Drupal",
    "nginx": "Nginx", "apache": "Apache", "php": "PHP",
    "python": "Python", "node.js": "Node.js", "java": "Java",
    "react": "React", "angular": "Angular", "vue": "Vue.js",
}

# ── Rx aliases used in UI (must be module-level for Prefab to detect) ─────────

VERSION          = Rx("version_display").default("PULSE —")
UPTIME           = Rx("uptime_display").default("up —")
RAM              = Rx("ram_display").default("RAM —/— GB")
TOOL_COUNT       = Rx("tools_display").default("— tools")
STATUS           = Rx("server_status").default("unknown")
STATUS_VARIANT   = Rx("server_status_variant").default("default")

# New header icons Rx
CPU_PCT          = Rx("cpu_display").default("0%")
RAM_DETAIL       = Rx("ram_detail_display").default("0/0 GB")
DISK_PCT         = Rx("disk_display").default("0%")
CPU_SPARK        = Rx("cpu_history").default([])
SCOPE_TARGET     = Rx("scope_target").default("No target scanned yet")
SCOPE_TYPE       = Rx("scope_type").default("unknown")
SCOPE_SUMMARY    = Rx("scope_summary").default("No scans yet")
SURFACE_TARGET   = Rx("surface_target").default("No target")
RISK_LEVEL       = Rx("risk_level").default("unknown")
RISK_VARIANT     = Rx("risk_variant").default("default")
PORTS_DISPLAY    = Rx("ports_display").default("No ports detected")
PORTS_COUNT      = Rx("ports_count").default(0)
SURFACE_PORTS    = Rx("surface_ports").default([])
SURFACE_TECHS    = Rx("surface_techs").default([])
FINDINGS         = Rx("findings").default([])
RECENT_SCANS     = Rx("recent_scans").default([])
INTELLIGENCE     = Rx("intelligence").default([])

rx_version       = VERSION
rx_uptime        = UPTIME
rx_ram           = RAM
rx_tools         = TOOL_COUNT
rx_status        = STATUS
rx_status_var    = STATUS_VARIANT
rx_cpu           = CPU_PCT
rx_ram_detail    = RAM_DETAIL
rx_disk          = DISK_PCT
rx_cpu_spark     = CPU_SPARK
rx_scope_target  = SCOPE_TARGET
rx_scope_type    = SCOPE_TYPE
rx_scope_summary = SCOPE_SUMMARY
rx_surface_target = SURFACE_TARGET
rx_risk          = RISK_LEVEL
rx_risk_var      = RISK_VARIANT
rx_ports_display = PORTS_DISPLAY
rx_surface_ports = SURFACE_PORTS
rx_surface_techs = SURFACE_TECHS
rx_findings      = FINDINGS
rx_recent_scans  = RECENT_SCANS

# Plan IDE
PLAN_STEPS   = Rx("plan_steps").default([])
PLAN_SUMMARY = Rx("plan_summary").default("No plan available")
rx_plan_summary = PLAN_SUMMARY
rx_plan_steps   = PLAN_STEPS

# Active Tools
ACTIVE_PROCS = Rx("active_processes").default(0)
ACTIVE_WORK  = Rx("active_workers").default(0)
ACTIVE_QUEUE = Rx("active_queue").default(0)
ACTIVE_SUM   = Rx("active_summary").default("No active tasks")
rx_active_procs  = ACTIVE_PROCS
rx_active_work   = ACTIVE_WORK
rx_active_queue  = ACTIVE_QUEUE
rx_active_sum    = ACTIVE_SUM

# History
HISTORY = Rx("history").default([])
rx_history = HISTORY

# Rate Limit
RL_PROFILE    = Rx("rl_profile").default("normal")
RL_CONFIDENCE = Rx("rl_confidence").default(0)
RL_EVENTS     = Rx("rl_events").default([])
RL_DELAY      = Rx("rl_delay").default(500)
RL_THREADS    = Rx("rl_threads").default(20)
RL_TIMEOUT    = Rx("rl_timeout").default(10)
RL_SUMMARY    = Rx("rl_summary").default("No rate limit events detected")

RL_CONF_DISP  = Rx("rl_confidence_display").default("0%")
RL_DELAY_DISP = Rx("rl_delay_display").default("Delay 500ms")

rx_rl_profile = RL_PROFILE
rx_rl_conf    = RL_CONFIDENCE
rx_rl_events  = RL_EVENTS
rx_rl_delay   = RL_DELAY
rx_rl_threads = RL_THREADS
rx_rl_to      = RL_TIMEOUT
rx_rl_summary = RL_SUMMARY
rx_rl_conf_display = RL_CONF_DISP
rx_rl_delay_display = RL_DELAY_DISP

# Errors & Failures
ERR_TOTAL    = Rx("error_total").default(0)
ERR_TOOL     = Rx("error_by_tool").default([])
ERR_TIMEOUTS = Rx("timeout_by_tool").default([])
ERR_SLOWEST  = Rx("slowest_tools").default([])
ERR_TYPE     = Rx("error_by_type").default([])
ERR_RECENT   = Rx("recent_errors").default([])
ERR_SR       = Rx("error_success_rate_display").default("0%")
ERR_SUM      = Rx("error_summary").default("No errors recorded")

rx_err_total   = ERR_TOTAL
rx_err_tool    = ERR_TOOL
rx_err_timeout = ERR_TIMEOUTS
rx_err_slowest = ERR_SLOWEST
rx_err_type    = ERR_TYPE
rx_err_recent  = ERR_RECENT
rx_err_sr      = ERR_SR
rx_err_sum     = ERR_SUM

# Badge variant for RL profile
_RL_VARIANTS = {"stealth": "destructive", "conservative": "warning", "normal": "default", "aggressive": "secondary"}
def _rl_variant(profile: str) -> str:
    return _RL_VARIANTS.get(profile, "default")
rx_rl_var = Rx("rl_variant").default("default")

# Footer
TOTAL_RUNS_DISP = Rx("total_runs_display").default("0 runs")
rx_runs      = TOTAL_RUNS_DISP

# Tool Performance
PERF_DATA     = Rx("tool_performance").default([])
PERF_TIMEOUTS = Rx("perf_timeouts").default([])
PERF_SUM      = Rx("perf_summary").default("No data")
rx_perf       = PERF_DATA
rx_perf_to    = PERF_TIMEOUTS
rx_perf_sum   = PERF_SUM

# Cache Status
CACHE_HITS  = Rx("cache_hits").default(0)
CACHE_MISS  = Rx("cache_misses").default(0)
CACHE_RATIO = Rx("cache_hit_ratio_display").default("\u2014%")
CACHE_SIZE  = Rx("cache_size").default(0)
CACHE_MAX   = Rx("cache_max_size").default(500)
CACHE_UTIL  = Rx("cache_util_display").default("0%")
CACHE_TOOL  = Rx("cache_by_tool").default([])
CACHE_SUM   = Rx("cache_summary_text").default("No cache data")
rx_cache_hits  = CACHE_HITS
rx_cache_miss  = CACHE_MISS
rx_cache_ratio = CACHE_RATIO
rx_cache_size  = CACHE_SIZE
rx_cache_max   = CACHE_MAX
rx_cache_util  = CACHE_UTIL
rx_cache_tool  = CACHE_TOOL
rx_cache_sum   = CACHE_SUM

# Cache Intelligence
CACHE_TTL    = Rx("cache_ttl_scores").default([])
CACHE_TTL_SUM= Rx("cache_ttl_summary").default("No TTL data")
rx_ttl_scores = CACHE_TTL
rx_ttl_sum    = CACHE_TTL_SUM

# System Trends
TREND_CPU_AVG  = Rx("trend_cpu_avg_display").default("0%")
TREND_MEM_AVG  = Rx("trend_mem_avg_display").default("0%")
TREND_PERIOD   = Rx("trend_period_display").default("\u2014")
TREND_MEASURES = Rx("trend_measurements").default(0)
TREND_CPU_HIST = Rx("trend_cpu_history").default([])
TREND_MEM_HIST = Rx("trend_mem_history").default([])
TREND_DISK     = Rx("trend_disk_display").default("0%")
rx_trend_cpu  = TREND_CPU_AVG
rx_trend_mem  = TREND_MEM_AVG
rx_trend_per  = TREND_PERIOD
rx_trend_meas = TREND_MEASURES
rx_trend_cpu_hist = TREND_CPU_HIST
rx_trend_mem_hist = TREND_MEM_HIST
rx_trend_disk = TREND_DISK

# Sessions
SESS_ACTIVE     = Rx("sessions_active").default([])
SESS_COMPLETED  = Rx("sessions_completed").default([])
SESS_SUM        = Rx("sessions_summary").default("No sessions")
rx_sess_active    = SESS_ACTIVE
rx_sess_completed = SESS_COMPLETED
rx_sess_sum       = SESS_SUM

# Confirmations
CONF_ACCEPTED = Rx("conf_accepted").default(0)
CONF_DENIED   = Rx("conf_denied").default(0)
CONF_SKIPPED  = Rx("conf_skipped").default(0)
CONF_SUM      = Rx("conf_summary").default("No confirmation events")
rx_conf_acc  = CONF_ACCEPTED
rx_conf_den  = CONF_DENIED
rx_conf_skip = CONF_SKIPPED
rx_conf_sum  = CONF_SUM

# Network I/O
NET_SENT    = Rx("net_sent_display").default("0 B")
NET_RECV    = Rx("net_recv_display").default("0 B")
NET_TOTAL   = Rx("net_total_display").default("0 B")
rx_net_sent  = NET_SENT
rx_net_recv  = NET_RECV
rx_net_total = NET_TOTAL

# Async scans
AS_RUNNING  = Rx("async_scans_running").default([])
AS_COMPLETE = Rx("async_scans_complete").default([])
AS_SUM      = Rx("async_scans_summary").default("No async scans")
rx_as_run  = AS_RUNNING
rx_as_done = AS_COMPLETE
rx_as_sum  = AS_SUM

# Missing tools
MISSING_TOOLS = Rx("missing_tools").default([])
MISSING_COUNT = Rx("missing_count").default(0)
rx_missing_tools = MISSING_TOOLS
rx_missing_count = MISSING_COUNT


# ═════════════════════════════════════════════════════════════════════════════
# Backend tools
# ═════════════════════════════════════════════════════════════════════════════

@app.tool(model=True)
def get_overview() -> dict:
    """PULSE dashboard overview: version, uptime, RAM, disk, CPU, tools count, server health.

    Call FIRST when starting a fresh session to discover the HexStrike environment.
    Returns system resources and server status — no target required.

    Returns: version_display, uptime_display, ram_display (avail/total GB),
    cpu_percent, cpu_history (for sparklines), disk_percent, tools_count,
    server_status ('healthy' or 'limited'), total_runs, total_errors.

    Example: get_overview()
    Next: get_scope() to see active target
    """
    summary = _op_metrics.summary()
    sys = summary.get("system", {})
    has_psutil = "cpu_percent" in sys
    avail = sys.get("memory_available_gb", 0)
    total = sys.get("memory_total_gb", 1)
    uptime = summary["uptime_seconds"]
    cpu = sys.get("cpu_percent", 0)
    disk = sys.get("disk_usage_percent", 0)

    cpu_history = []
    try:
        rm = enhanced_process_manager.resource_monitor
        recent = rm.usage_history[-10:] if hasattr(rm, "usage_history") else []
        cpu_history = [u.get("cpu_percent", 0) for u in recent if "cpu_percent" in u]
    except Exception:
        pass

    return {
        "version":               app_config["VERSION"],
        "uptime_seconds":        uptime,
        "ram_percent":           sys.get("memory_percent", 0),
        "ram_available_gb":      avail,
        "ram_total_gb":          total,
        "disk_free_gb":          sys.get("disk_free_gb", 0),
        "disk_percent":          disk,
        "cpu_percent":           cpu,
        "cpu_history":           cpu_history,
        "server_status":         "healthy" if has_psutil else "limited",
        "tools_count":           len(TOOLS),
        "total_runs":            summary["total_runs"],
        "total_errors":          summary["total_errors"],
        # Pre-formatted display strings
        "version_display":       f"PULSE v{app_config['VERSION']}",
        "uptime_display":        f"up {_fmt_duration(uptime)}",
        "ram_display":           f"RAM {avail}/{total} GB",
        "tools_display":         f"{len(TOOLS)} tools",
        "cpu_display":           f"{cpu:.0f}%",
        "ram_detail_display":    f"{avail}/{total} GB",
        "disk_display":          f"{disk:.0f}%",
        "server_status_variant": "success" if has_psutil else "warning",
    }


def _guess_target_type(target_str: str) -> str:
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", target_str):
        return "ip"
    if re.match(r"^[\w.-]+\.[a-z]{2,}(:\d+)?$", target_str):
        return "domain"
    if target_str.startswith(("http://", "https://")):
        return "url"
    return "unknown"

@app.tool()
def get_scope(target: str | None = None) -> dict:
    """Detect current scope from recent scan cache entries or explicit target."""
    now = time.time()

    # Explicit target takes precedence
    if target:
        return {
            "active_target":  target,
            "target_type":    _guess_target_type(target),
            "tools_used":     [],
            "tools_count":    0,
            "last_seen_ago":  0,
            "last_tool":      None,
            "age_seconds":    0,
            "scope_summary":  "Target set via pulse command",
        }

    try:
        entries = sorted(
            _scan_cache.values(),
            key=lambda v: v.get("timestamp", 0),
            reverse=True,
        )[:100]
    except Exception:
        return {"active_target": None}

    targets: dict = {}
    for e in entries:
        t = e.get("target", "")
        if not t:
            continue
        if t not in targets:
            targets[t] = {
                "target": t,
                "tools": set(),
                "first_seen": e.get("timestamp", 0),
                "last_seen": e.get("timestamp", 0),
            }
        targets[t]["tools"].add(e.get("tool", "?"))
        targets[t]["first_seen"] = min(
            targets[t]["first_seen"], e.get("timestamp", 0))
        targets[t]["last_seen"] = max(
            targets[t]["last_seen"], e.get("timestamp", 0))

    if not targets:
        return {"active_target": None}

    active = max(targets.values(), key=lambda x: x["last_seen"])
    target_str = active["target"]
    target_type = _guess_target_type(target_str)

    all_tools = sorted(active["tools"])
    last_tool_entry = next(
        (e for e in entries if e.get("target") == active["target"]),
        None,
    )

    ago_sec = now - active["last_seen"]
    age_sec = now - active["first_seen"]
    return {
        "active_target":  active["target"],
        "target_type":    target_type,
        "tools_used":     [{"name": t} for t in all_tools],
        "tools_count":    len(all_tools),
        "last_seen_ago":  ago_sec,
        "last_tool":      last_tool_entry.get("tool") if last_tool_entry else None,
        "age_seconds":    age_sec,
        "scope_summary": (
            f"Last scan: {_fmt_duration(ago_sec)} ago"
            f"  \u00b7  {len(all_tools)} tools"
            f"  \u00b7  Active {_fmt_duration(age_sec)}"
        ),
    }


@app.tool(model=True)
def get_surface(target: str | None = None) -> dict:
    """Open ports, services, and technology detection for a target.

    Parses cached nmap ports + whatweb tech detections. Auto-uses active
    scope target if none provided. Risk level: high (>5 ports), medium (>2),
    low (>0), unknown.

    Returns: target, risk_level, ports (list with port/service/state),
    port_count, technologies (list), ports_display, risk_variant.

    Use AFTER get_scope() to assess attack surface.
    Example: get_surface() — uses current scope target
    Example: get_surface('scanme.nmap.org')
    Next: get_findings() for vulnerabilities
    """
    if not target:
        scope = get_scope()
        target = scope.get("active_target")
    if not target:
        return {"target": None}

    entries = _cache_for_target(target)

    ports = []
    for e in entries:
        if e.get("tool") not in ("nmap", "nmap_advanced"):
            continue
        result = e.get("result", {})
        output = str(result.get("output", "") or result.get("stdout", ""))
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and "/" in parts[0] and parts[1] == "open":
                try:
                    port = int(parts[0].split("/")[0])
                    service = parts[2] if len(parts) >= 3 else ""
                    if port not in (p["port"] for p in ports):
                        ports.append({"port": port, "service": service, "state": "open"})
                except ValueError:
                    pass

    techs = set()
    for e in entries:
        if e.get("tool") != "whatweb":
            continue
        result = e.get("result", {})
        output = str(result.get("output", "") or result.get("stdout", "")).lower()
        for keyword, label in _TECH_KEYWORDS.items():
            if keyword in output:
                techs.add(label)

    port_count = len(ports)
    if port_count > 5:
        risk = "high"
    elif port_count > 2:
        risk = "medium"
    elif port_count > 0:
        risk = "low"
    else:
        risk = "unknown"

    suggestion = _suggest_next_from_context({
        "ports": sorted(ports, key=lambda p: p["port"]),
        "technologies": sorted(techs),
    }, [])
    result = {
        "target":         target,
        "ports":          sorted(ports, key=lambda p: p["port"]),
        "ports_count":    port_count,
        "technologies":   sorted(techs),
        "risk_level":     risk,
        "risk_variant":   "destructive" if risk == "high" else "warning" if risk == "medium" else "default",
        "ports_display":  f"{port_count} open port{'s' if port_count != 1 else ''}" if port_count else "No ports detected",
    }
    if suggestion:
        result["next_suggested_tool"] = suggestion
    return result


@app.tool(model=True)
def get_findings(target: str | None = None) -> list[dict]:
    """Vulnerabilities and issues for a target from nuclei + nikto scan cache.

    Returns findings sorted by severity (critical→high→medium→low→info).
    Each finding: tool, severity, finding (ID or URL), details.
    Auto-uses active scope target if none provided.

    Use AFTER get_surface() to find actual vulnerabilities.
    Example: get_findings()
    Example: get_findings('scanme.nmap.org')
    Next: get_plan() for attack chain
    """
    if not target:
        scope = get_scope()
        target = scope.get("active_target")
    if not target:
        return []

    entries = _cache_for_target(target)
    findings = []

    for e in entries:
        tool = e.get("tool", "")
        result = e.get("result", {})
        raw = str(result.get("output", "") or result.get("stdout", ""))
        output = _strip_ansi(raw)

        if tool == "nuclei":
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                sev_m = re.search(
                    r"\[(critical|high|medium|low|info)\]",
                    line, re.IGNORECASE,
                )
                if sev_m:
                    sev = sev_m.group(1).lower()
                    brackets = re.findall(r"\[([^\]]*)\]", line)
                    finding_id = brackets[1] if len(brackets) >= 2 else ""
                    parts = line.split()
                    url = parts[-1] if parts else ""
                    findings.append({
                        "tool": tool,
                        "severity": sev,
                        "finding": finding_id,
                        "details": url[:120],
                    })

        elif tool == "nikto":
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("+ /"):
                    findings.append({
                        "tool": tool,
                        "severity": "info",
                        "finding": line[2:].strip()[:100],
                        "details": "",
                    })

    # Enrich with Couche 1 + Layer 2 (pure functions, idempotent)
    for f in findings:
        exploit = suggest_exploit(f)
        if exploit:
            f["exploit"] = exploit
        f["layer2"] = compute_layer2_score(f)
        score = f["layer2"]["score"]
        f["score"] = f"{score:.2f}" if score > 0 else "—"

    findings.sort(key=lambda f: f.get("layer2", {}).get("score", 0), reverse=True)
    return findings


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
def get_rate_limit_status(target: str | None = None) -> dict:
    """Rate limit detection state per target. Shows current profile, confidence, indicators."""
    events = list(_rate_limit_events)
    if target:
        events = [e for e in events if e.get("target") == target]
    recent = events[-1] if events else None
    return {
        "profile":        recent["profile"] if recent else "normal",
        "confidence":     recent["confidence"] if recent else 0.0,
        "indicators":     recent["indicators"] if recent else [],
        "event_count":    len(events),
        "events":         events[-5:],
        "last_detected":  recent["timestamp"] if recent else None,
        "timing":         _RL_PROFILES.get(recent["profile"] if recent else "normal", {}),
    }


_RL_PROFILES = {
    "aggressive":   {"delay": 0.1, "threads": 50, "timeout": 5},
    "normal":       {"delay": 0.5, "threads": 20, "timeout": 10},
    "conservative": {"delay": 1.0, "threads": 10, "timeout": 15},
    "stealth":      {"delay": 2.0, "threads": 5,  "timeout": 30},
}


@app.tool()
def get_errors_and_failures() -> dict:
    """Error and failure statistics. Per-tool error/timeout counts, slowest tools, error type distribution, recent errors."""
    ops = _op_metrics.summary()
    error_by_tool = _op_metrics.error_count_by_tool()
    timeout_by_tool = _op_metrics.timeout_count_by_tool()
    slowest = _op_metrics.slowest_tools(10)
    success_rate = _op_metrics.success_rate_by_tool()

    err_stats = {}
    try:
        err_stats = error_handler.get_error_statistics()
    except Exception:
        pass

    recent_errors = []
    for e in err_stats.get("recent_errors", []):
        recent_errors.append({
            "tool": e.get("tool", "?"),
            "type": e.get("error_type", "unknown"),
            "ts":   str(e.get("timestamp", ""))[:19],
        })

    error_type_list = [
        {"type": t.replace("_", " ").title(), "count": c}
        for t, c in err_stats.get("error_counts_by_type", {}).items()
    ]
    error_type_list.sort(key=lambda x: -x["count"])

    for e in error_by_tool:
        e["display"] = f"{e['errors']}/{e['runs']}"
    for e in slowest:
        e["avg_display"] = _fmt_duration(e.get("avg_duration"))
        e["max_display"] = _fmt_duration(e.get("max_duration"))
    for e in timeout_by_tool:
        e["display"] = f"{e['timeouts']}/{e['runs']}"
    for e in success_rate:
        e["rate_display"] = f"{int(e['success_rate'] * 100)}%"

    return {
        "total_errors":         ops.get("total_errors", 0),
        "total_runs":           ops.get("total_runs", 0),
        "global_success_rate":  ops.get("global_success_rate", 0),
        "error_by_tool":        error_by_tool[:10],
        "timeout_by_tool":      timeout_by_tool[:10],
        "slowest_tools":        slowest[:10],
        "success_rate_by_tool": success_rate[:10],
        "error_by_type":        error_type_list,
        "recent_errors":        recent_errors[-10:],
    }


@app.tool(model=True)
def get_plan(target: str | None = None, objective: str = "comprehensive") -> dict:
    """Attack chain for a target from the IntelligentDecisionEngine.

    Generates ordered steps with tool, expected outcome, success probability,
    and estimated time. Auto-uses active scope target if none provided.

    Returns: target, steps (list with num/tool/outcome/probability/ETA),
    step_count, estimated_time, risk_level, summary.

    Use AFTER get_findings() to plan the attack workflow.
    Example: get_plan()
    Example: get_plan('scanme.nmap.org')
    """
    if not target:
        scope = get_scope()
        target = scope.get("active_target")
    if not target:
        return {"target": None, "steps": [], "step_count": 0, "summary": "No target"}
    try:
        ide = get_decision_engine()
        profile = ide.analyze_target(target)
        chain = ide.create_attack_chain(profile, objective)
        data = chain.to_dict()
        for i, step in enumerate(data.get("steps", []), 1):
            step["num"] = i
            prob = step.get("success_probability", 0)
            step["prob_display"] = f"{int(prob * 100)}%"
            step["eta_display"] = _fmt_duration(step.get("execution_time_estimate", 0))
            outcome = step.get("expected_outcome", "")
            step["outcome_short"] = outcome[:60] + "..." if len(outcome) > 60 else outcome
        data["step_count"] = len(data.get("steps", []))
        data["summary"] = (
            f"{data['step_count']} steps \u00b7 "
            f"{_fmt_duration(data['estimated_time'])} est \u00b7 "
            f"{data['risk_level']} risk"
        )
        return data
    except Exception as e:
        return {
            "target": target, "steps": [], "step_count": 0,
            "summary": f"Plan unavailable: {str(e)[:80]}",
        }


@app.tool()
def get_active_tools() -> dict:
    """Currently running / active processes from process manager."""
    try:
        stats = enhanced_process_manager.get_comprehensive_stats()
        pool = stats.get("process_pool", {})
        active = stats.get("active_processes", 0)
        workers = pool.get("active_workers", 0)
        queue = pool.get("queue_size", 0)
        resource = stats.get("resource_usage", {})
        return {
            "active_processes":  active,
            "active_workers":    workers,
            "queue_size":        queue,
            "pool_stats":        pool,
            "resource_usage":    resource,
            "auto_scaling":      stats.get("auto_scaling_enabled", False),
            "summary":           f"{active} active \u00b7 {workers} workers \u00b7 {queue} queued",
        }
    except Exception as e:
        return {
            "active_processes": 0, "active_workers": 0, "queue_size": 0,
            "summary": f"Unavailable: {str(e)[:80]}",
        }


@app.tool()
def get_history(target: str | None = None, limit: int = 50) -> list[dict]:
    """Scan history, optionally filtered by target. Replaces get_recent_scans."""
    now = time.time()
    try:
        entries = sorted(
            _scan_cache.values(),
            key=lambda v: v.get("timestamp", 0),
            reverse=True,
        )
    except Exception:
        return []

    if target:
        entries = [e for e in entries if e.get("target") == target]

    result = []
    for e in entries[:limit]:
        r = e.get("result", {})
        exec_time = r.get("execution_time")
        result.append({
            "tool":              e.get("tool", "?"),
            "target":            e.get("target", "?"),
            "timestamp":         e.get("timestamp", 0),
            "age":               _fmt_duration(now - e.get("timestamp", 0)) if e.get("timestamp") else "\u2014",
            "status":            "\u2713" if r.get("success") else "\u2717",
            "execution_time":    round(exec_time, 1) if exec_time else None,
            "execution_display": _fmt_duration(exec_time) if exec_time else "\u2014",
            "error":             (r.get("error", "") or "")[:80],
        })
    return result


@app.tool()
def get_tool_performance() -> dict:
    """Per-tool success rates and timeout counts. Shows worst performers first."""
    sr = _op_metrics.success_rate_by_tool()
    to = _op_metrics.timeout_count_by_tool()
    to_map = {e["tool"]: e["timeouts"] for e in to}

    combined = []
    for e in sr:
        combined.append({
            "tool":      e["tool"],
            "runs":      e["runs"],
            "successes": e["successes"],
            "errors":    e["errors"],
            "rate_display": f"{int(e['success_rate'] * 100)}%",
            "timeouts":  to_map.get(e["tool"], 0),
        })
    summary = f"{len(sr)} tools \u00b7 best: {sr[-1]['tool'] if sr else '--'} {int(sr[-1]['success_rate'] * 100) if sr else 0}%"

    for e in to:
        e["display"] = f"{e['timeouts']}/{e['runs']}"

    return {
        "tools":    combined,
        "timeouts": to,
        "summary":  summary,
    }


@app.tool()
def get_cache_status() -> dict:
    """Cache hit/miss statistics and per-tool cache hits."""
    cs = _op_metrics.cache_summary()
    tool_hits = _op_metrics.cache_hits_by_tool()

    adv_stats = {}
    try:
        from server_core.singletons import cache
        adv_stats = cache.get_stats()
    except Exception:
        pass

    return {
        "hits":        cs.get("hits", 0),
        "misses":      cs.get("misses", 0),
        "total":       cs.get("total", 0),
        "hit_ratio":   cs.get("hit_ratio", 0),
        "cache_size":  adv_stats.get("size", 0),
        "max_size":    adv_stats.get("max_size", 500),
        "hit_rate":    adv_stats.get("hit_rate", "0%"),
        "utilization": adv_stats.get("utilization", 0),
        "by_tool":     tool_hits,
    }


def get_cache_intelligence() -> dict:
    """Per-tool adaptive TTL learning statistics."""
    try:
        from mcp_core.server_setup import _scan_cache
        ttl_scores = _scan_cache.get_ttl_scores()
    except Exception:
        return {"scores": [], "summary": "No TTL data"}

    rows = []
    for tool, info in sorted(ttl_scores.items()):
        hit_ratio = info.get("hit_ratio", 0)
        hit_ratio_display = f"{hit_ratio * 100:.0f}%" if hit_ratio else "0%"
        ttl_seconds = int(info.get("current_ttl_seconds", 1800))
        ttl_display = _fmt_duration(ttl_seconds)
        rows.append({
            "tool": tool,
            "hits": info.get("hits", 0),
            "misses": info.get("misses", 0),
            "hit_ratio_display": hit_ratio_display,
            "current_ttl_display": ttl_display,
        })

    ttl_range = f"{rows[0]['current_ttl_display']}\u2013{rows[-1]['current_ttl_display']}" if rows else "N/A"
    summary = f"{len(rows)} tools tracked \u00b7 TTL range {ttl_range}"
    return {"scores": rows, "summary": summary}


@app.tool()
def get_system_trends() -> dict:
    """System resource trends over time. CPU/memory averages, history for sparklines."""
    try:
        rm = enhanced_process_manager.resource_monitor
        trends = rm.get_usage_trends() if hasattr(rm, "get_usage_trends") else {}
        history = list(rm.usage_history) if hasattr(rm, "usage_history") else []
    except Exception:
        return {
            "cpu_avg": 0, "memory_avg": 0, "measurements": 0,
            "period_minutes": 0, "cpu_history": [], "mem_history": [],
            "disk_display": "0%",
        }

    cpu_hist = [h["cpu_percent"] for h in history[-30:] if "cpu_percent" in h]
    mem_hist = [h["memory_percent"] for h in history[-30:] if "memory_percent" in h]
    disk = history[-1].get("disk_percent", 0) if history else 0

    return {
        "cpu_avg":         trends.get("cpu_avg_10", 0),
        "memory_avg":      trends.get("memory_avg_10", 0),
        "measurements":    trends.get("measurements", len(history)),
        "period_minutes":  trends.get("trend_period_minutes", 0),
        "cpu_history":     cpu_hist,
        "mem_history":     mem_hist,
        "disk_display": f"{int(disk)}%",
    }


@app.tool()
def get_sessions() -> dict:
    """Active and completed session summaries from SessionStore.

    Returns counts and lists of recent sessions for the dashboard.
    No target required — shows all sessions across targets.

    Returns: active_count, completed_count, active (list of session IDs),
    completed (list with session_id, target, findings, tools_executed, timestamps).
    """
    try:
        from server_core.singletons import get_session_store
        ss = get_session_store()
        active = ss.list_active()
        completed = ss.list_completed()
    except Exception as e:
        return {
            "active_count": 0, "completed_count": 0,
            "active": [], "completed": [],
            "summary": f"Unavailable: {str(e)[:60]}",
        }

    recent = completed[:20]
    for s in recent:
        s["tools_str"] = ", ".join(s.get("tools_executed", [])[:5])
        s["age_display"] = _fmt_duration(time.time() - s.get("updated_at", 0)) if s.get("updated_at") else "\u2014"

    summary = f"{len(active)} active \u00b7 {len(completed)} completed" if completed else f"{len(active)} active"
    return {
        "active_count":    len(active),
        "completed_count": len(completed),
        "active":          active[-10:],
        "completed":       recent,
        "summary":         summary,
    }


@app.tool()
def get_confirmations() -> dict:
    """Confirmation event statistics (accepted/denied/skipped).

    Tracks user confirmations for dangerous operations.
    No target required — global statistics.

    Returns: accepted, denied, skipped, total, summary.
    """
    conf = _op_metrics.confirmation_summary()
    total = sum(conf.values())
    summary_parts = []
    if conf.get("accepted"):
        summary_parts.append(f"{conf['accepted']} accepted")
    if conf.get("denied"):
        summary_parts.append(f"{conf['denied']} denied")
    if conf.get("skipped"):
        summary_parts.append(f"{conf['skipped']} skipped")
    summary = " \u00b7 ".join(summary_parts) if summary_parts else "No confirmation events"
    return {
        "accepted": conf.get("accepted", 0),
        "denied":   conf.get("denied", 0),
        "skipped":  conf.get("skipped", 0),
        "total":    total,
        "summary":  summary,
    }


@app.tool()
def get_network_io() -> dict:
    """Network I/O statistics (bytes sent/received) from ResourceMonitor.

    Shows current cumulative counters and rate estimation.
    No target required — system-wide network stats.

    Returns: bytes_sent, bytes_recv, bytes_sent_display, bytes_recv_display,
    total_display.
    """
    try:
        rm = enhanced_process_manager.resource_monitor
        history = list(rm.usage_history) if hasattr(rm, "usage_history") else []
    except Exception:
        return {
            "bytes_sent": 0, "bytes_recv": 0,
            "bytes_sent_display": "0 B", "bytes_recv_display": "0 B",
            "total_display": "0 B",
        }

    if not history:
        return {
            "bytes_sent": 0, "bytes_recv": 0,
            "bytes_sent_display": "0 B", "bytes_recv_display": "0 B",
            "total_display": "0 B",
        }

    latest = history[-1]
    sent = latest.get("network_bytes_sent", 0)
    recv = latest.get("network_bytes_recv", 0)

    def _fmt_bytes(b):
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    return {
        "bytes_sent":          sent,
        "bytes_recv":          recv,
        "bytes_sent_display":  _fmt_bytes(sent),
        "bytes_recv_display":  _fmt_bytes(recv),
        "total_display":       _fmt_bytes(sent + recv),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Async scan execution — Phase 3 (fix timeout 300s)
# ═════════════════════════════════════════════════════════════════════════════

from mcp_core.server_setup import get_direct_tools

_async_scans: dict = {}
_async_scans_lock = threading.Lock()


def _cleanup_old_scans(max_age: float = 3600):
    """Remove scans older than max_age seconds."""
    now = time.time()
    with _async_scans_lock:
        stale = [sid for sid, s in _async_scans.items()
                 if s.get("end_time", now) and now - s.get("end_time", now) > max_age]
        for sid in stale:
            del _async_scans[sid]


@app.tool()
def run_async_tool(tool: str = "nmap", target: str = "", params: str = "") -> dict:
    """Run a security tool in background. Returns immediately with scan_id.

    Use for long-running tools (sqlmap, nikto, nuclei, nmap full port scans)
    that would exceed Claude Desktop's stdio timeout (~300s).

    The scan runs on the server in background. Poll status with get_scan_status(scan_id).
    Completed results also appear in get_history().

    Args:
        tool: Tool name (nmap, sqlmap, whatweb, nikto, nuclei, gobuster, etc.)
        target: Target hostname, IP, or URL
        params: JSON string of tool parameters, e.g. '{"ports":"80,443","scan_type":"-sV"}'

    Returns:
        dict with scan_id (str), status ("started"), tool, target
    """
    scan_id = f"scan_{int(time.time())}_{os.urandom(4).hex()}"

    parsed_params: dict = {}
    if params:
        try:
            parsed_params = json.loads(params)
        except json.JSONDecodeError:
            return {"scan_id": None, "status": "error", "error": "Invalid JSON in params"}

    # Ensure target is in params
    if target:
        parsed_params["target"] = target

    now = time.time()
    with _async_scans_lock:
        _async_scans[scan_id] = {
            "tool": tool, "target": target, "status": "starting",
            "start_time": now, "end_time": None,
            "result": None, "progress": 0, "error": None,
        }

    def _run():
        try:
            entry = get_direct_tools().get(tool)
            if not entry:
                raise ValueError(f"Unknown tool: {tool}")

            exec_func, binary_name = entry

            with _async_scans_lock:
                _async_scans[scan_id]["status"] = "running"

            start = time.time()
            result = exec_func(tool, parsed_params)
            elapsed = time.time() - start

            with _async_scans_lock:
                _async_scans[scan_id].update({
                    "status": "completed",
                    "end_time": time.time(),
                    "result": {
                        "success": result.get("success", False),
                        "stdout": (result.get("output", "") or "")[:2000],
                        "execution_time": elapsed,
                        "error": (result.get("error", "") or "")[:200],
                        "returncode": result.get("returncode", -1),
                    },
                })

            try:
                _op_metrics.record({
                    "tool": tool, "success": result.get("success", False),
                    "duration": elapsed, "timed_out": False, "cache_hit": False,
                })
            except Exception:
                pass

        except Exception as e:
            with _async_scans_lock:
                _async_scans[scan_id].update({
                    "status": "failed", "end_time": time.time(),
                    "error": str(e)[:200],
                })

    threading.Thread(target=_run, daemon=True, name=f"async-{tool}").start()
    _cleanup_old_scans()

    return {"scan_id": scan_id, "status": "started", "tool": tool, "target": target}


@app.tool()
def get_scan_status(scan_id: str) -> dict:
    """Poll status of an async scan started by run_async_tool.

    Returns current status, elapsed time, and result when complete.
    For completed scans, result includes execution_time, returncode, error (no full stdout).

    Args:
        scan_id: The scan_id returned by run_async_tool()

    Returns:
        dict with scan_id, status (started/running/completed/failed/not_found),
        tool, target, elapsed (seconds), result (if completed), error (if failed)
    """
    with _async_scans_lock:
        entry = _async_scans.get(scan_id)

    if not entry:
        return {"scan_id": scan_id, "status": "not_found"}

    elapsed = round(time.time() - entry["start_time"], 1)
    if entry.get("end_time"):
        elapsed = round(entry["end_time"] - entry["start_time"], 1)

    result = {
        "scan_id": scan_id,
        "status": entry["status"],
        "tool": entry["tool"],
        "target": entry["target"],
        "elapsed": elapsed,
        "elapsed_display": _fmt_duration(elapsed),
    }

    if entry["result"]:
        r = entry["result"]
        result["result"] = {
            "success": r.get("success", False),
            "execution_time": r.get("execution_time", 0),
            "error": r.get("error", ""),
            "returncode": r.get("returncode", -1),
        }
        stdout = r.get("stdout", "")
        if stdout:
            result["stdout_preview"] = stdout[:500]

    if entry.get("error"):
        result["error"] = entry["error"]

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard state collector — shared between UI and live dashboard tool
# ═════════════════════════════════════════════════════════════════════════════

TOOLS_BY_INTENSITY = {
    "quick":  ["nmap", "whatweb"],
    "medium": ["nmap", "whatweb", "nuclei", "nikto"],
    "full":   ["nmap", "whatweb", "nuclei", "nikto", "gobuster"],
}

# Tools that need url=http:// prefix instead of target= (direct IP doesn't work)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


_TOOLS_NEED_URL = {"whatweb", "gobuster", "sqlmap", "wpscan", "dalfox", "jaeles", "xsser"}
_TOOLS_NEED_URL_AS_TARGET = {"nuclei", "httpx", "katana"}
_TOOLS_NEED_HOST = {"nmap", "nmap_advanced"}


def _suggest_next_from_context(surface: dict, findings: list) -> dict:
    """Suggest next tool based on structured surface + findings data.

    Primary signal: highest layer2.score ≥ 0.3 with exploit tool.
    Fallback: keyword/severity-based on findings, then surface-based.

    Returns dict with 'tool', 'reason', 'expected_time', 'priority'.
    Empty dict if context is insufficient.
    Priority levels: critical (score ≥ 0.5) > high (score ≥ 0.3) > medium (probable) > low (exploratory).
    """
    _EST = ESTIMATED_TIMES

    # ── Score-aware suggestion ────────────────────────────────────────────
    if findings:
        top_finding = None
        top_score = -1.0
        top_actionable = None
        top_actionable_score = -1.0
        for f in findings:
            if not isinstance(f, dict):
                continue
            l2 = f.get("layer2", {})
            score = l2.get("score", 0) if isinstance(l2, dict) else 0
            exploit = f.get("exploit", {})
            tool = exploit.get("tool", "") if isinstance(exploit, dict) else ""
            if score >= 0.3 and tool and score > top_score:
                top_score = score
                top_finding = f
            if score >= 0.3 and tool and tool != "manual" and score > top_actionable_score:
                top_actionable_score = score
                top_actionable = f

        best = top_actionable if top_actionable else top_finding
        if best:
            ex = best.get("exploit", {})
            tool = ex.get("tool", "") if isinstance(ex, dict) else ""
            finding_text = str(best.get("finding", ""))
            details = str(best.get("details", ""))[:60]
            best_score = top_actionable_score if best is top_actionable else top_score
            priority = "critical" if best_score >= 0.5 else "high"

            return {
                "tool": tool,
                "reason": f"{finding_text} (score {best_score:.3f}) — {details}",
                "expected_time": _EST.get(tool, "1-10 min"),
                "priority": priority,
            }

    # ── Fallback: keyword/severity-based on findings ──────────────────────
    if findings:
        findings_severities = []
        findings_text = []
        for f in findings:
            if isinstance(f, dict):
                findings_severities.append(str(f.get("severity", "")).lower())
                findings_text.append(str(f.get("finding", "")).lower())

        findings_all = " ".join(findings_text)
        has_critical = any(s == "critical" for s in findings_severities)
        has_high = any(s == "high" for s in findings_severities)

        if "sql" in findings_all or "sqli" in findings_all or "injection" in findings_all:
            return {"tool": "sqlmap", "reason": "SQL injection candidate found — confirm and exploit", "expected_time": _EST.get("sqlmap", "2-30 min"), "priority": "critical"}
        if "xss" in findings_all or "cross-site" in findings_all:
            return {"tool": "dalfox", "reason": "XSS candidate found — validate with dalfox", "expected_time": _EST.get("dalfox", "1-5 min"), "priority": "critical"}
        if "smb" in findings_all or "eternalblue" in findings_all or "ms17" in findings_all:
            return {"tool": "metasploit", "reason": "SMB vulnerability confirmed — attempt exploitation", "expected_time": _EST.get("metasploit", "1-5 min"), "priority": "critical"}
        if "ssl" in findings_all or "tls" in findings_all or "certificate" in findings_all:
            return {"tool": "testssl", "reason": "SSL/TLS issues reported — deep inspection", "expected_time": _EST.get("testssl", "30-60s"), "priority": "high"}
        # Severity-based metasploit shortcut: only if layer2 scores are absent
        # (when scores are present, the score-aware path above already decided)
        has_any_score = any(
            isinstance(f.get("layer2"), dict) and "score" in f["layer2"]
            for f in findings if isinstance(f, dict)
        )
        if (has_critical or has_high) and not has_any_score:
            return {"tool": "metasploit", "reason": "Critical/high severity findings — attempt exploitation", "expected_time": _EST.get("metasploit", "1-5 min"), "priority": "high"}

    # ── Fallback: surface-based suggestions ──────────────────────────────
    ports = surface.get("ports", []) if isinstance(surface, dict) else []
    port_numbers = {p.get("port") for p in ports if isinstance(p, dict)}
    services = [str(p.get("service", "")).lower() for p in ports if isinstance(p, dict)]
    services_str = " ".join(services)

    techs = surface.get("technologies", []) if isinstance(surface, dict) else []
    techs_lower = [t.lower() for t in techs if isinstance(t, str)]

    if port_numbers:
        if 80 in port_numbers or 443 in port_numbers or 8080 in port_numbers:
            if any("wordpress" in t for t in techs_lower):
                return {"tool": "wpscan", "reason": "WordPress detected — enumerate plugins/users", "expected_time": _EST.get("wpscan", "1-10 min"), "priority": "high"}
            if any("joomla" in t for t in techs_lower):
                return {"tool": "joomscan", "reason": "Joomla detected — enumerate extensions", "expected_time": _EST.get("joomscan", "1-5 min"), "priority": "high"}
            if techs_lower:
                return {"tool": "gobuster", "reason": "Web server detected with tech — discover hidden paths", "expected_time": "1-5 min", "priority": "high"}
            return {"tool": "whatweb", "reason": "Web ports open — identify technologies", "expected_time": "10-30s", "priority": "high"}
        if 445 in port_numbers:
            return {"tool": "smbmap", "reason": "SMB port 445 open — enumerate shares", "expected_time": _EST.get("smbmap", "10-30s"), "priority": "high"}
        if 22 in port_numbers:
            return {"tool": "hydra", "reason": "SSH port 22 open — test credentials", "expected_time": _EST.get("hydra", "5-30 min"), "priority": "medium"}
        if 1433 in port_numbers or 3306 in port_numbers or 5432 in port_numbers or 27017 in port_numbers:
            return {"tool": "sqlmap", "reason": "Database port open — test for weak auth", "expected_time": _EST.get("sqlmap", "2-30 min"), "priority": "medium"}
        if "smb" in services_str or "microsoft-ds" in services_str:
            return {"tool": "smbmap", "reason": "SMB service detected — enumerate shares", "expected_time": _EST.get("smbmap", "10-30s"), "priority": "high"}
        if "http" in services_str or "ssl" in services_str:
            return {"tool": "whatweb", "reason": "Web service detected — fingerprint technologies", "expected_time": "10-30s", "priority": "high"}

    # Fallback: low-severity findings with no port context
    if findings:
        return {"tool": "gobuster", "reason": "Findings reviewed — continue with directory discovery", "expected_time": "1-5 min", "priority": "low"}

    # No context
    if port_numbers:
        return {"tool": "nuclei", "reason": "Ports discovered — run vulnerability scan", "expected_time": "1-5 min", "priority": "medium"}

    return {}

def _collect_dashboard_state(target: str | None = None) -> dict:
    """Collect all dashboard data sources into a flat state dict.

    Shared between pulse_dashboard() (UI) and get_live_dashboard() (tool).
    """
    overview = get_overview()
    scope = get_scope(target) if target else get_scope()
    active_target = scope.get("active_target")
    surface = get_surface(active_target) if active_target else {"target": None}
    findings = get_findings(active_target) if active_target else []
    plan = get_plan(active_target) if active_target else {"target": None, "steps": [], "step_count": 0, "summary": "No target"}
    active = get_active_tools()
    history = get_history(active_target)
    rl = get_rate_limit_status(active_target)
    rl_events_table = [
        {
            "tool":       e.get("tool", ""),
            "target":     e.get("target", ""),
            "profile":    e.get("profile", ""),
            "indicators": ", ".join(e.get("indicators", []))[:80],
        }
        for e in rl.get("events", [])
    ]
    sys = _op_metrics.summary().get("system", {})
    ops = _op_metrics.summary()
    total_runs_display = f"{ops['total_runs']} runs"

    err = get_errors_and_failures()
    err_sr = err.get("global_success_rate", 0)
    error_summary = (
        f"{err.get('total_errors', 0)} errors \u00b7 "
        f"{len(err.get('timeout_by_tool', []))} tools with timeouts \u00b7 "
        f"{int(err_sr * 100)}% success"
    )
    error_success_rate_display = f"{int(err_sr * 100)}%" if err.get("total_runs", 0) > 0 else "\u2014"

    perf = get_tool_performance()
    cache_status = get_cache_status()
    trends = get_system_trends()
    sessions = get_sessions()
    confirmations = get_confirmations()
    netio = get_network_io()

    # Async scans data for panel
    with _async_scans_lock:
        now = time.time()
        running_list = [
            {"scan_id": sid, "tool": s["tool"], "target": s["target"],
             "elapsed": _fmt_duration(now - s["start_time"]),
             "status": s["status"]}
            for sid, s in _async_scans.items()
            if s["status"] in ("starting", "running")
        ][-10:]
        complete_list = [
            {"scan_id": sid, "tool": s["tool"], "target": s["target"],
             "elapsed": _fmt_duration(s.get("end_time", now) - s["start_time"]),
             "status": s["status"]}
            for sid, s in _async_scans.items()
            if s["status"] in ("completed", "failed")
        ][-20:]
    async_running_count = len(running_list)
    async_complete_count = len(complete_list)
    async_scans_summary = (
        f"{async_running_count} running \u00b7 {async_complete_count} completed"
        if async_running_count or async_complete_count
        else "No async scans"
    )

    cache_hit_ratio = cache_status.get("hit_ratio", 0)
    cache_hit_ratio_display = f"{int(cache_hit_ratio * 100)}%" if cache_status.get("total", 0) > 0 else "\u2014"
    cache_util = cache_status.get("utilization", 0)
    cache_util_display = f"{int(cache_util)}%" if cache_util else "0%"
    cache_summary_text = (
        f"{cache_status.get('hits', 0)} hits \u00b7 {cache_status.get('misses', 0)} misses \u00b7 "
        f"{cache_status.get('cache_size', 0)}/{cache_status.get('max_size', 500)} entries"
    )
    # Cache Intelligence — adaptive TTL scores
    cache_ttl_raw = get_cache_intelligence()
    cache_ttl_scores = cache_ttl_raw.get("scores", [])
    cache_ttl_summary = cache_ttl_raw.get("summary", "No TTL data")
    trend_cpu_avg = trends.get("cpu_avg", 0)
    trend_mem_avg = trends.get("memory_avg", 0)
    trend_cpu_avg_display = f"{int(trend_cpu_avg)}%"
    trend_mem_avg_display = f"{int(trend_mem_avg)}%"
    trend_period_display = f"{trends.get('period_minutes', 0):.1f}m" if trends.get("period_minutes", 0) else "\u2014"

    # TargetStore — automatically record scans for MCP Resources
    try:
        ts = get_target_store()
        if active_target and history:
            ts.record_scan(
                target=active_target,
                tools_used=list({h.get("tool", "?") for h in history}),
            )
    except Exception:
        pass

    return {
        # Raw data objects
        "overview":       overview,
        "scope":          scope,
        "active_target":  active_target,
        "surface":        surface,
        "findings":       findings,
        "plan":           plan,
        "active":         active,
        "history":        history,
        "rl":             rl,
        "rl_events_table": rl_events_table,
        "sys":            sys,
        "ops":            ops,
        "err":            err,
        "perf":           perf,
        "cache_status":   cache_status,
        "trends":         trends,
        "sessions":       sessions,
        "confirmations":  confirmations,
        "netio":          netio,
        # Display helpers
        "total_runs_display":    total_runs_display,
        "error_summary":         error_summary,
        "error_success_rate_display": error_success_rate_display,
        "cache_hit_ratio_display": cache_hit_ratio_display,
        "cache_util_display":      cache_util_display,
        "cache_summary_text":      cache_summary_text,
        "trend_cpu_avg_display": trend_cpu_avg_display,
        "trend_mem_avg_display": trend_mem_avg_display,
        "trend_period_display":  trend_period_display,
        "async_scans_running":  running_list,
        "async_scans_complete": complete_list,
        "async_scans_summary":  async_scans_summary,
        "missing_tools": _registry.get_missing(),
        "cache_ttl_scores":  cache_ttl_scores,
        "cache_ttl_summary": cache_ttl_summary,
        "next_suggested_tool": _suggest_next_from_context(surface, findings) if active_target else {},
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — Live dashboard (single-call for Claude, replaces 15+ get_* calls)
# ═════════════════════════════════════════════════════════════════════════════

@app.tool(model=True)
def get_live_dashboard(target: str | None = None) -> dict:
    """Full Pulse dashboard state in one call. Returns all panels: Overview, Scope, Surface, Findings, Plan, Active Tools, History, Rate Limit, Errors & Failures, Tool Performance, Cache Status, System Trends, Sessions, Confirmations, Network I/O, Async Scans, Intelligence.

    No target needed for system-wide stats. Pass target to filter scope/surface/findings/history/plan to a specific host. Use this INSTEAD of calling 15+ individual get_* tools — one call gives you everything. ~100ms typical response.

    Returns {overview, scope, surface, findings, plan, active_tools, history, rate_limit, errors, tool_performance, cache_status, system_trends, sessions, confirmations, network_io, async_scans, intelligence} with pre-formatted display strings.
    """
    st = _collect_dashboard_state(target)
    return {
        "overview":         st["overview"],
        "scope":            st["scope"],
        "surface":          st["surface"],
        "findings":         st["findings"],
        "plan":             st["plan"],
        "active_tools":     st["active"],
        "history":          st["history"],
        "rate_limit":       st["rl"],
        "errors":           st["err"],
        "tool_performance": st["perf"],
        "cache_status":     st["cache_status"],
        "system_trends":    st["trends"],
        "sessions":         st["sessions"],
        "confirmations":    st["confirmations"],
        "network_io":       st["netio"],
        "async_scans": {
            "running":  st["async_scans_running"],
            "complete": st["async_scans_complete"],
            "summary":  st["async_scans_summary"],
        },
        "intelligence":     get_tool_intelligence(),
        "next_suggested_tool": st.get("next_suggested_tool", {}),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Unified scan entry point
# ═════════════════════════════════════════════════════════════════════════════

@app.tool(model=True)
def scan(target: str = "", intensity: str = "quick", objective: str = "comprehensive") -> dict:
    """Full reconnaissance scan on a target. Runs appropriate security tools based on intensity, then returns surface analysis + vulnerability findings + attack plan in one response.

    Intensity levels:
    - quick (default): nmap + whatweb — open ports and technology detection. ~30s when uncached.
    - medium: + nuclei + nikto — adds vulnerability scanning. ~2-3 min.
    - full: + gobuster (web targets) — complete recon. ~5-10 min.

    Uses scan cache — recently scanned targets return instantly.

    For scans that may take >30s (intensity=medium/full, CIDR ranges, multiple tools),
    use scan_background() instead — it returns a task_id immediately and runs in the
    background, allowing the agent to continue working while the scan progresses.

    Returns {target, intensity, tools, surface, findings, plan, summary}.
    Pass `objective` to guide the attack planner (default: comprehensive).
    """
    # Resolve target
    scope_data = get_scope(target) if target else get_scope()
    resolved = scope_data.get("active_target") or target
    if not resolved:
        return {"error": "No target specified or found in scope", "target": None, "surface": None, "findings": [], "plan": None}

    intensity = str(intensity).lower()
    if intensity not in TOOLS_BY_INTENSITY:
        intensity = "quick"

    tools_to_run = TOOLS_BY_INTENSITY[intensity]
    _direct = get_direct_tools()
    tool_results = {}

    _workers = min(len(tools_to_run), 5)
    with ThreadPoolExecutor(max_workers=_workers) as pool:
        futures = {pool.submit(_run_scan_tool, name, resolved, _direct): name
                   for name in tools_to_run}
        for future in as_completed(futures):
            tr = future.result()
            tool_results[tr.pop("tool_name")] = tr

    surface_data = get_surface(resolved)
    findings_data = get_findings(resolved) if intensity in ("medium", "full") else []
    # Enrichment Couche 1 + Layer 2 now inside get_findings() — no duplicate needed here
    plan_data = get_plan(resolved, objective) if intensity == "full" else {"target": resolved, "steps": [], "step_count": 0, "summary": "Skipped — use full intensity for planning"}

    # TargetStore record for MCP Resources
    try:
        ts = get_target_store()
        ts.record_scan(
            target=resolved,
            tools_used=list(tool_results.keys()),
            surface_data=surface_data,
            findings=findings_data,
        )
    except Exception:
        pass

    suggestion = _suggest_next_from_context(surface_data, findings_data)

    # ── cache_age ────────────────────────────────────────────────────────
    tool_statuses = {t: v.get("status") for t, v in tool_results.items()}
    n_cached = sum(1 for s in tool_statuses.values() if s == "cached")
    n_completed = sum(1 for s in tool_statuses.values() if s == "completed")
    n_failed = sum(1 for s in tool_statuses.values() if s in ("failed", "error", "timeout"))
    total_tools = len(tool_results)

    if n_cached > 0 and n_completed == 0:
        cache_entries = _cache_for_target(resolved)
        cached_times = [e.get("timestamp", 0) for e in cache_entries
                        if e.get("tool") in tool_results]
        max_age_sec = time.time() - (max(cached_times) if cached_times else time.time())
        if max_age_sec < 60:
            freshness = "fresh_cache"
            age_str = f"{max_age_sec:.0f}s ago"
        elif max_age_sec < 3600:
            freshness = "recent_cache"
            age_str = f"{int(max_age_sec // 60)} min ago"
        else:
            freshness = "stale_cache"
            age_str = f"{int(max_age_sec // 3600)}h ago"
    else:
        freshness = "fresh_scan"
        age_str = "just now"

    # ── highlights ───────────────────────────────────────────────────────
    highlights = [
        {
            "severity": f["severity"],
            "tool": f.get("tool"),
            "finding": f.get("finding", "")[:80],
            "exploit": f.get("exploit", {}),
        }
        for f in findings_data[:3] if isinstance(f, dict)
    ]

    # ── severity breakdown ───────────────────────────────────────────────
    sev_counts = {}
    for f in findings_data:
        if isinstance(f, dict):
            s = f.get("severity", "info")
            sev_counts[s] = sev_counts.get(s, 0) + 1
    sev_display = ", ".join(f"{c} {s}" for s in ("critical", "high", "medium", "low", "info")
                           if (c := sev_counts.get(s)))

    return {
        "target":    resolved,
        "intensity": intensity,
        "tools":     tool_results,
        "surface":   surface_data,
        "findings":  findings_data,
        "plan":      plan_data,
        "cache_age": {
            "status": freshness,
            "age": age_str,
            "age_seconds": max_age_sec if n_cached > 0 and n_completed == 0 else 0,
            "tools_cached": n_cached,
            "tools_completed": n_completed,
            "tools_failed": n_failed,
        },
        "highlights": highlights,
        "summary": (
            f"{'cached' if freshness.startswith('fresh_cache') or freshness.startswith('recent_cache') or freshness.startswith('stale_cache') else 'fresh'} scan "
            f"({age_str.replace('just now', 'parallel').replace('s ago', 's').replace(' min ago', 'm').replace('h ago', 'h')})"
            f" · {total_tools} tools ({n_completed} ok"
            + (f", {n_failed} fail" if n_failed else ", 0 fail")
            + f")"
            f" · {surface_data.get('ports_count', 0)} ports"
            + (f" · {sev_display}" if sev_display else "")
            + (f" · next: {suggestion.get('tool')}" if suggestion else "")
        ),
    } | ({"next_suggested_tool": suggestion} if suggestion else {})


# ═════════════════════════════════════════════════════════════════════════════
# UI entry point
# ═════════════════════════════════════════════════════════════════════════════

@app.ui()
def pulse_dashboard() -> PrefabApp:
    """Open the Pulse dashboard (3+2+1 grid layout)."""
    st = _collect_dashboard_state()
    overview = st["overview"]
    scope = st["scope"]
    active_target = st["active_target"]
    surface = st["surface"]
    findings = st["findings"]
    plan = st["plan"]
    active = st["active"]
    history = st["history"]
    rl = st["rl"]
    rl_events_table = st["rl_events_table"]
    sys = st["sys"]
    ops = st["ops"]
    err = st["err"]
    perf = st["perf"]
    cache_status = st["cache_status"]
    trends = st["trends"]
    sessions = st["sessions"]
    confirmations = st["confirmations"]
    netio = st["netio"]

    total_runs_display = st["total_runs_display"]
    error_summary = st["error_summary"]
    error_success_rate_display = st["error_success_rate_display"]
    cache_hit_ratio_display = st["cache_hit_ratio_display"]
    cache_util_display = st["cache_util_display"]
    cache_summary_text = st["cache_summary_text"]
    cache_ttl_scores = st["cache_ttl_scores"]
    cache_ttl_summary = st["cache_ttl_summary"]
    trend_cpu_avg_display = st["trend_cpu_avg_display"]
    trend_mem_avg_display = st["trend_mem_avg_display"]
    trend_period_display = st["trend_period_display"]
    running_list = st["async_scans_running"]
    complete_list = st["async_scans_complete"]
    async_scans_summary = st["async_scans_summary"]

    with Column(gap=0) as view:

        # ── Header ─────────────────────────────────────────────────────
        cpu_has_history = len(overview.get("cpu_history", [])) > 1
        with Column(gap=0):
            with Row(gap=2, align="center", css_class="p-2 px-4 border-b flex-wrap"):
                with Tooltip(content=f"{overview['cpu_display']} CPU", side="bottom"):
                    with Row(gap=1, align="center"):
                        Icon(name="cpu", size="sm")
                        Progress(value=rx_cpu, variant="default", css_class="w-12")
                Text("\u00b7", css_class="text-xs text-muted")
                with Tooltip(content=f"RAM {overview['ram_detail_display']}", side="bottom"):
                    with Row(gap=1, align="center"):
                        Icon(name="hard-drive", size="sm")
                        Text(f"{rx_ram_detail}", css_class="text-xs font-mono")
                Text("\u00b7", css_class="text-xs text-muted")
                with Tooltip(content=f"Disk {overview['disk_display']}", side="bottom"):
                    with Row(gap=1, align="center"):
                        Icon(name="database", size="sm")
                        Progress(value=rx_disk, variant="default", css_class="w-12")
                Div(css_class="flex-1")
                Text(f"{rx_version}", css_class="text-xs font-bold tracking-wider")
                Div(css_class="flex-1")
                Badge(f"{rx_tools}", variant="outline")
            if cpu_has_history:
                with Row(css_class="px-4 py-0.5 border-b bg-muted/5"):
                    Sparkline(data=rx_cpu_spark, height=16, variant="info", fill=True, curve="smooth")

        # ── Scope bar ──────────────────────────────────────────────────
        with Row(gap=3, align="center", css_class="p-2 px-4 bg-muted/30 border-b flex-wrap"):
            Muted("SCOPE")
            Text(f"{rx_scope_target}", css_class="font-bold")
            Badge(f"{rx_scope_type}", variant="outline")
            Muted(f"{rx_scope_summary}")

        # ── Grid 3 colonnes: Surface | Findings | Plan IDE ─────────────
        with Row(gap=4, css_class="p-4 items-start"):
            with Column(gap=2, css_class="flex-1"):
                Muted("SURFACE", css_class="text-xs uppercase tracking-wider")
                with Card():
                    with CardContent(css_class="p-3"):
                        with Column(gap=2):
                            with Row(gap=2, align="center"):
                                Badge(f"{rx_risk}", variant=rx_risk_var)
                                Text(f"{rx_ports_display}", css_class="text-sm")
                            with Row(gap=1, css_class="flex-wrap"):
                                with ForEach("surface_ports") as p:
                                    Badge(p.service or str(p.port), variant="outline")
                            with Row(gap=1, css_class="flex-wrap"):
                                with ForEach("surface_techs") as t:
                                    Badge(t, variant="secondary")

            with Column(gap=2, css_class="flex-1"):
                Muted("FINDINGS", css_class="text-xs uppercase tracking-wider")
                DataTable(
                    columns=[
                        DataTableColumn(key="severity", header="Sev"),
                        DataTableColumn(key="finding",  header="Finding"),
                        DataTableColumn(key="score",    header="Sc."),
                        DataTableColumn(key="tool",     header="Tool"),
                        DataTableColumn(key="details",  header="Details"),
                    ],
                    rows=Rx("findings"),
                )

            with Column(gap=2, css_class="flex-1"):
                Muted("PLAN IDE", css_class="text-xs uppercase tracking-wider")
                Text(f"{rx_plan_summary}", css_class="text-sm text-muted")
                DataTable(
                    columns=[
                        DataTableColumn(key="num",          header="#"),
                        DataTableColumn(key="tool",         header="Tool"),
                        DataTableColumn(key="outcome_short", header="Outcome"),
                        DataTableColumn(key="prob_display", header="Prob"),
                        DataTableColumn(key="eta_display",  header="ETA"),
                    ],
                    rows=Rx("plan_steps"),
                )

        Separator()

        # ── Grid 2 colonnes: Historique | Active Tools ─────────────────
        with Row(gap=4, css_class="p-4 items-start"):
            with Column(gap=2, css_class="flex-1"):
                Muted("HISTORY", css_class="text-xs uppercase tracking-wider")
                DataTable(
                    columns=[
                        DataTableColumn(key="tool",              header="Tool"),
                        DataTableColumn(key="target",            header="Target"),
                        DataTableColumn(key="age",               header="When"),
                        DataTableColumn(key="status",            header="\u2713"),
                        DataTableColumn(key="execution_display", header="Time"),
                    ],
                    rows=Rx("history"),
                )

            with Column(gap=2, css_class="flex-1"):
                Muted("ACTIVE TOOLS", css_class="text-xs uppercase tracking-wider")
                with Card():
                    with CardContent(css_class="p-3"):
                        with Column(gap=2):
                            with Row(gap=4):
                                Metric(label="Processes", value=Rx("active_processes"))
                                Metric(label="Workers",   value=Rx("active_workers"))
                                Metric(label="Queued",    value=Rx("active_queue"))
                            Text(f"{rx_active_sum}", css_class="text-sm text-muted")

                Separator(css_class="my-1")
                Muted("ASYNC SCANS", css_class="text-xs")
                with Column(gap=1, css_class="max-h-[200px] overflow-y-auto"):
                    Muted(f"{rx_as_sum}", css_class="text-xs text-muted")
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",    header="Tool"),
                            DataTableColumn(key="target",  header="Target"),
                            DataTableColumn(key="elapsed", header="Time"),
                            DataTableColumn(key="status",  header="Status"),
                        ],
                        rows=Rx("async_scans_running"),
                    )
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",    header="Tool"),
                            DataTableColumn(key="target",  header="Target"),
                            DataTableColumn(key="elapsed", header="Time"),
                            DataTableColumn(key="status",  header="Status"),
                        ],
                        rows=Rx("async_scans_complete"),
                    )

        Separator()

        # ── Rate Limit ──────────────────────────────────────────────────
        Muted("RATE LIMIT", css_class="text-xs uppercase tracking-wider p-4")
        with Row(gap=4, css_class="px-4 pb-4 items-start flex-wrap"):
            with Card():
                with CardContent(css_class="p-3"):
                    with Column(gap=2):
                        with Row(gap=3, align="center"):
                            Badge(f"{rx_rl_profile}", variant=rx_rl_var)
                            Muted(f"{rx_rl_conf_display}")
                        with Row(gap=2, css_class="flex-wrap"):
                            Muted(f"{rx_rl_delay_display}")
                            Muted(f"\u00b7")
                            Muted(f"{rx_rl_threads} threads")
                            Muted(f"\u00b7")
                            Muted(f"timeout {rx_rl_to}s")
                        Muted(f"{rx_rl_summary}", css_class="text-sm text-muted")
            with Card(css_class="flex-1"):
                with CardContent(css_class="p-3"):
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",     header="Tool"),
                            DataTableColumn(key="target",   header="Target"),
                            DataTableColumn(key="profile",  header="Profile"),
                            DataTableColumn(key="indicators", header="Triggers"),
                        ],
                        rows=Rx("rl_events"),
                    )

        Separator()

        # ── Errors & Failures ────────────────────────────────────────────
        Muted("ERRORS & FAILURES", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            with Row(gap=3, align="center"):
                Badge(f"{rx_err_total} errors", variant="destructive")
                Muted(f"{rx_err_sr} success")
                Muted(f"{rx_err_sum}")
            with Row(gap=4, css_class="items-start flex-wrap"):
                with Column(gap=2, css_class="flex-1 min-w-[200px]"):
                    Muted("By tool", css_class="text-xs")
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",    header="Tool"),
                            DataTableColumn(key="display", header="Err/Runs"),
                        ],
                        rows=Rx("error_by_tool"),
                    )
                with Column(gap=2, css_class="flex-1 min-w-[200px]"):
                    Muted("Timeouts", css_class="text-xs")
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",    header="Tool"),
                            DataTableColumn(key="display", header="To/Runs"),
                        ],
                        rows=Rx("timeout_by_tool"),
                    )
                with Column(gap=2, css_class="flex-1 min-w-[200px]"):
                    Muted("Slowest tools", css_class="text-xs")
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",        header="Tool"),
                            DataTableColumn(key="avg_display", header="Avg"),
                            DataTableColumn(key="max_display", header="Max"),
                            DataTableColumn(key="runs",        header="Runs"),
                        ],
                        rows=Rx("slowest_tools"),
                    )
            with Row(gap=2, css_class="pt-2"):
                Muted("By error type", css_class="text-xs")
            DataTable(
                columns=[
                    DataTableColumn(key="type",  header="Error Type"),
                    DataTableColumn(key="count", header="Count"),
                ],
                rows=Rx("error_by_type"),
            )
            with Row(gap=2, css_class="pt-2"):
                Muted("Recent errors", css_class="text-xs")
            DataTable(
                columns=[
                    DataTableColumn(key="tool", header="Tool"),
                    DataTableColumn(key="type", header="Type"),
                    DataTableColumn(key="ts",   header="Timestamp"),
                ],
                rows=Rx("recent_errors"),
            )

        Separator()

        # ── Tool Performance ────────────────────────────────────────────
        Muted("TOOL PERFORMANCE", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            Muted(f"{rx_perf_sum}", css_class="text-sm text-muted")
            with Row(gap=4, css_class="items-start flex-wrap"):
                with Column(gap=2, css_class="flex-1 min-w-[200px]"):
                    Muted("Success rate", css_class="text-xs")
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",        header="Tool"),
                            DataTableColumn(key="rate_display", header="Rate"),
                            DataTableColumn(key="runs",        header="Runs"),
                            DataTableColumn(key="timeouts",    header="To"),
                        ],
                        rows=Rx("tool_performance"),
                    )
                with Column(gap=2, css_class="flex-1 min-w-[200px]"):
                    Muted("Tools with timeouts", css_class="text-xs")
                    DataTable(
                        columns=[
                            DataTableColumn(key="tool",    header="Tool"),
                            DataTableColumn(key="display", header="To/Runs"),
                        ],
                        rows=Rx("perf_timeouts"),
                    )

        Separator()

        # ── Missing Tools ────────────────────────────────────────────────
        Muted("MISSING TOOLS", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            with Row(gap=3, align="center"):
                Badge(f"{rx_missing_count} missing", variant="warning")
                Muted("tools without binary on PATH — use install_tool()", css_class="text-sm text-muted")
            DataTable(
                columns=[
                    DataTableColumn(key="name",  header="Tool"),
                    DataTableColumn(key="binary", header="Binary"),
                    DataTableColumn(key="category", header="Category"),
                    DataTableColumn(key="install_hint", header="Install hint"),
                ],
                rows=Rx("missing_tools"),
            )

        Separator()

        # ── Cache Status ────────────────────────────────────────────────
        Muted("CACHE STATUS", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            with Row(gap=4, css_class="flex-wrap"):
                with Card():
                    with CardContent(css_class="p-3"):
                        with Row(gap=4):
                            Metric(label="Hits", value=Rx("cache_hits"))
                            Metric(label="Misses", value=Rx("cache_misses"))
                            Metric(label="Hit ratio", value=Rx("cache_hit_ratio_display"))
                            Metric(label="Size", value=Rx("cache_size"))
                            Metric(label="Max", value=Rx("cache_max_size"))
                            Metric(label="Util", value=Rx("cache_util_display"))
            Muted(f"{rx_cache_sum}", css_class="text-sm text-muted pt-1")
            DataTable(
                columns=[
                    DataTableColumn(key="tool",       header="Tool"),
                    DataTableColumn(key="cache_hits", header="Cache hits"),
                    DataTableColumn(key="runs",       header="Runs"),
                ],
                rows=Rx("cache_by_tool"),
            )

        Separator()

        # ── Cache Intelligence ───────────────────────────────────────────
        Muted("CACHE INTELLIGENCE", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            Muted(Rx("cache_ttl_summary"), css_class="text-sm text-muted")
            DataTable(
                columns=[
                    DataTableColumn(key="tool",               header="Tool"),
                    DataTableColumn(key="hits",               header="Hits"),
                    DataTableColumn(key="misses",             header="Misses"),
                    DataTableColumn(key="hit_ratio_display",  header="Hit ratio"),
                    DataTableColumn(key="current_ttl_display",header="TTL"),
                ],
                rows=Rx("cache_ttl_scores"),
            )

        Separator()

        # ── System Trends ───────────────────────────────────────────────
        Muted("SYSTEM TRENDS", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            with Row(gap=4, css_class="flex-wrap"):
                with Card():
                    with CardContent(css_class="p-3"):
                        with Row(gap=4):
                            Metric(label="CPU avg", value=Rx("trend_cpu_avg_display"))
                            Metric(label="MEM avg", value=Rx("trend_mem_avg_display"))
                            Metric(label="Period", value=Rx("trend_period_display"))
                            Metric(label="Measures", value=Rx("trend_measurements"))
            with Row(gap=4, css_class="pt-2 flex-wrap"):
                with Column(gap=1, css_class="flex-1 min-w-[200px]"):
                    Muted("CPU", css_class="text-xs")
                    Sparkline(data=Rx("trend_cpu_history"), height=24, variant="info", fill=True, curve="smooth")
                with Column(gap=1, css_class="flex-1 min-w-[200px]"):
                    Muted("Memory", css_class="text-xs")
                    Sparkline(data=Rx("trend_mem_history"), height=24, variant="warning", fill=True, curve="smooth")

        Separator()

        # ── Sessions ────────────────────────────────────────────────────
        Muted("SESSIONS", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            Muted(f"{rx_sess_sum}", css_class="text-sm text-muted")
            with Row(gap=4, css_class="items-start flex-wrap"):
                with Column(gap=2, css_class="flex-1 min-w-[200px]"):
                    Muted("Completed", css_class="text-xs")
                    DataTable(
                        columns=[
                            DataTableColumn(key="session_id",   header="Session"),
                            DataTableColumn(key="target",       header="Target"),
                            DataTableColumn(key="total_findings", header="Finds"),
                            DataTableColumn(key="age_display",  header="Age"),
                        ],
                        rows=Rx("sessions_completed"),
                    )

        Separator()

        # ── Confirmations ───────────────────────────────────────────────
        Muted("CONFIRMATIONS", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            with Card():
                with CardContent(css_class="p-3"):
                    with Row(gap=4):
                        Metric(label="Accepted", value=Rx("conf_accepted"))
                        Metric(label="Denied",   value=Rx("conf_denied"))
                        Metric(label="Skipped",  value=Rx("conf_skipped"))
            Muted(f"{rx_conf_sum}", css_class="text-sm text-muted pt-1")

        Separator()

        # ── Network I/O ─────────────────────────────────────────────────
        Muted("NETWORK I/O", css_class="text-xs uppercase tracking-wider p-4")
        with Column(gap=2, css_class="px-4 pb-4"):
            with Card():
                with CardContent(css_class="p-3"):
                    with Row(gap=4):
                        Metric(label="Sent",     value=Rx("net_sent_display"))
                        Metric(label="Received", value=Rx("net_recv_display"))
                        Metric(label="Total",    value=Rx("net_total_display"))

        Separator()

        # ── Intelligence ───────────────────────────────────────────────
        Muted("INTELLIGENCE", css_class="text-xs uppercase tracking-wider p-4")
        DataTable(
            columns=[
                DataTableColumn(key="tool",     header="Tool"),
                DataTableColumn(key="baseline", header="Baseline"),
                DataTableColumn(key="live",     header="Live"),
                DataTableColumn(key="blended",  header="Blended"),
                DataTableColumn(key="runs",     header="Runs"),
            ],
            rows=Rx("intelligence"),
        )

        Separator()

        # ── Footer ─────────────────────────────────────────────────────
        with Row(gap=4, align="center", css_class="p-2 px-4 bg-muted/20 border-t flex-wrap"):
            Muted(f"{rx_version}")
            Muted(f"{rx_runs}")

    return PrefabApp(
        view=view,
        state={
            # Overview
            "version":               overview["version"],
            "version_display":       overview["version_display"],
            "uptime_display":        overview["uptime_display"],
            "ram_display":           overview["ram_display"],
            "tools_display":         overview["tools_display"],
            "uptime_seconds":        overview["uptime_seconds"],
            "ram_percent":           overview["ram_percent"],
            "ram_available_gb":      overview["ram_available_gb"],
            "ram_total_gb":          overview["ram_total_gb"],
            "disk_percent":          overview["disk_percent"],
            "cpu_percent":           overview["cpu_percent"],
            "cpu_history":           overview.get("cpu_history", []),
            "cpu_display":           overview.get("cpu_display", "0%"),
            "ram_detail_display":    overview.get("ram_detail_display", "0/0 GB"),
            "disk_display":          overview.get("disk_display", "0%"),
            "server_status":         overview["server_status"],
            "server_status_variant": overview["server_status_variant"],
            "tools_count":           overview["tools_count"],
            "total_runs":            overview["total_runs"],
            "total_errors":          overview["total_errors"],
            # Footer stats
            "total_runs_display":    total_runs_display,
            # Scope
            "scope_target":          scope.get("active_target"),
            "scope_type":            scope.get("target_type"),
            "scope_tools":           scope.get("tools_used", []),
            "scope_tools_count":     scope.get("tools_count", 0),
            "scope_last_seen_ago":   scope.get("last_seen_ago"),
            "scope_age":             scope.get("age_seconds"),
            "scope_summary":         scope.get("scope_summary"),
            # Surface
            "surface_target":        surface.get("target"),
            "risk_level":            surface.get("risk_level", "unknown"),
            "risk_variant":          surface.get("risk_variant", "default"),
            "ports_display":         surface.get("ports_display", "No ports detected"),
            "ports_count":           surface.get("ports_count", 0),
            "surface_ports":         surface.get("ports", []),
            "surface_techs":         surface.get("technologies", []),
            # Findings
            "findings":              findings,
            # System resources
            "system":                sys,
            # Plan IDE
            "plan_target":           plan.get("target"),
            "plan_steps":            plan.get("steps", []),
            "plan_summary":          plan.get("summary", "No plan available"),
            # Active Tools
            "active_processes":      active.get("active_processes", 0),
            "active_workers":        active.get("active_workers", 0),
            "active_queue":          active.get("queue_size", 0),
            "active_summary":        active.get("summary", "No active tasks"),
            # History
            "history":               history,
            # Rate Limit
            "rl_profile":            rl.get("profile", "normal"),
            "rl_variant":            _rl_variant(rl.get("profile", "normal")),
            "rl_confidence":         rl.get("confidence", 0),
            "rl_delay":              rl.get("timing", {}).get("delay", 0.5),
            "rl_threads":            rl.get("timing", {}).get("threads", 20),
            "rl_timeout":            rl.get("timing", {}).get("timeout", 10),
            "rl_summary":            _fmt_rl_summary(rl),
            "rl_confidence_display": f"{int(rl.get('confidence', 0) * 100)}%",
            "rl_delay_display":      f"Delay {int(rl.get('timing', {}).get('delay', 0.5) * 1000)}ms",
            "rl_events":             rl_events_table,
            # Errors & Failures
            "error_total":               err.get("total_errors", 0),
            "error_success_rate_display": error_success_rate_display,
            "error_summary":             error_summary,
            "error_by_tool":             err.get("error_by_tool", []),
            "timeout_by_tool":           err.get("timeout_by_tool", []),
            "slowest_tools":             err.get("slowest_tools", []),
            "error_by_type":             err.get("error_by_type", []),
            "recent_errors":             err.get("recent_errors", []),
            # Tool Performance
            "tool_performance":   perf.get("tools", []),
            "perf_timeouts":      perf.get("timeouts", []),
            "perf_summary":       perf.get("summary", "No data"),
            # Missing Tools
            "missing_tools":  st.get("missing_tools", []),
            "missing_count":  len(st.get("missing_tools", [])),
            # Cache Status
            "cache_hits":              cache_status.get("hits", 0),
            "cache_misses":            cache_status.get("misses", 0),
            "cache_hit_ratio_display": cache_hit_ratio_display,
            "cache_size":              cache_status.get("cache_size", 0),
            "cache_max_size":          cache_status.get("max_size", 500),
            "cache_util_display":      cache_util_display,
            "cache_summary_text":      cache_summary_text,
            "cache_by_tool":           cache_status.get("by_tool", []),
            # Cache Intelligence
            "cache_ttl_scores":  cache_ttl_scores,
            "cache_ttl_summary": cache_ttl_summary,
            # System Trends
            "trend_cpu_avg_display": trend_cpu_avg_display,
            "trend_mem_avg_display": trend_mem_avg_display,
            "trend_period_display":  trend_period_display,
            "trend_measurements":    trends.get("measurements", 0),
            "trend_cpu_history":     trends.get("cpu_history", []),
            "trend_mem_history":     trends.get("mem_history", []),
            "trend_disk_display":    trends.get("disk_display", "0%"),
            # Sessions
            "sessions_active":    sessions.get("active", []),
            "sessions_completed": sessions.get("completed", []),
            "sessions_summary":   sessions.get("summary", "No sessions"),
            # Confirmations
            "conf_accepted": confirmations.get("accepted", 0),
            "conf_denied":   confirmations.get("denied", 0),
            "conf_skipped":  confirmations.get("skipped", 0),
            "conf_summary":  confirmations.get("summary", "No confirmation events"),
            # Network I/O
            "net_sent_display":  netio.get("bytes_sent_display", "0 B"),
            "net_recv_display":  netio.get("bytes_recv_display", "0 B"),
            "net_total_display": netio.get("total_display", "0 B"),
            # Async scans
            "async_scans_running":  running_list,
            "async_scans_complete": complete_list,
            "async_scans_summary":  async_scans_summary,
            # Intelligence
            "intelligence":          get_tool_intelligence(),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cache_for_target(target: str) -> list[dict]:
    """Return all scan cache entries for a given target."""
    try:
        return [v for v in _scan_cache.values() if v.get("target") == target]
    except Exception:
        return []


def _run_scan_tool(tool_name: str, resolved: str, _direct: dict) -> dict:
    """Run a single scan tool. Thread-safe — no shared state beyond _scan_cache (which is RLock-guarded).

    Returns {tool_name, status, error?, returncode?}.
    """
    cache_entries = _cache_for_target(resolved)
    cached = [c for c in cache_entries if str(c.get("tool", "")).lower() == tool_name]

    if cached:
        if tool_name in _TOOLS_NEED_HOST and cached[0].get("result", {}).get("stdout", "").startswith("Starting Nmap") and "0 hosts up" in cached[0].get("result", {}).get("stdout", ""):
            pass
        else:
            return {"tool_name": tool_name, "status": "cached", "cached": True}

    entry = _direct.get(tool_name)
    if not entry:
        return {"tool_name": tool_name, "status": "skipped", "error": f"Unknown tool: {tool_name}"}

    exec_func, binary = entry
    try:
        params = {"target": resolved}
        if tool_name in _TOOLS_NEED_URL:
            if not resolved.startswith(("http://", "https://")):
                params = {"url": f"http://{resolved}"}
            else:
                params = {"url": resolved}
        elif tool_name in _TOOLS_NEED_URL_AS_TARGET:
            if not resolved.startswith(("http://", "https://")):
                params = {"url": f"http://{resolved}", "target": resolved}
            else:
                params = {"url": resolved, "target": resolved}
        elif tool_name in _TOOLS_NEED_HOST:
            host = urlparse(resolved).hostname or resolved
            params = {"target": host, "scan_type": "-sTV"}

        # Parameter optimizer: adds --host-timeout, --timeout, threads, additional_args
        # Additive only — caller_keys protected, never overrides scan() built params
        params = _optimizer.optimize(tool_name, params)

        out = exec_func(binary, params)
        ok = out.get("success", False)
        result = {
            "tool_name": tool_name,
            "status": "completed" if ok else "failed",
            "returncode": out.get("returncode"),
        }
        if not ok:
            result["error"] = out.get("error", "Unknown error")

        stdout_str = out.get("stdout", "") or out.get("output", "")
        _scan_cache.set(f"{tool_name}:{uuid.uuid4().hex[:8]}", {
            "tool": tool_name,
            "target": resolved,
            "timestamp": time.time(),
            "status": "completed" if ok else "failed",
            "result": {
                "success": ok,
                "stdout": stdout_str,
                "output": stdout_str,
            },
        })
        return result
    except Exception as e:
        return {"tool_name": tool_name, "status": "error", "error": str(e)}


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "\u2014"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_rl_summary(rl: dict) -> str:
    """Format rate limit summary line."""
    n = rl.get("event_count", 0)
    if n == 0:
        return "No rate limit events detected"
    last = rl.get("last_detected")
    if last:
        ago = _fmt_duration(time.time() - last)
        return f"{n} event(s) \u00b7 Last {ago} ago"
    return f"{n} event(s)"


# ═════════════════════════════════════════════════════════════════════════════
# @app.ui() — CTF Dashboard
# ═════════════════════════════════════════════════════════════════════════════

@app.ui()
def ctf_dashboard() -> PrefabApp:
    """CTF challenge tracker — categories, tools, progress."""
    cm = get_ctf_manager()
    cats = cm.category_tools if hasattr(cm, "category_tools") else {}
    cats = cats or {}
    categories = sorted(cats.keys())
    _cat_tool_list = {
        c: [t for subs in (cats.get(c, {}) or {}).values() for t in subs]
        for c in categories
    }
    all_stats = _registry.get_all_stats() if hasattr(_registry, "get_all_stats") else {}

    cat_chart = [
        {"category": c, "tools": len(_cat_tool_list.get(c, []))}
        for c in categories
    ]

    with PrefabApp() as app:
        with Column(gap=4, css_class="p-4"):
            with Row(gap=3, align="center"):
                Icon(name="swords", size="default")
                Heading("CTF Challenge Dashboard", css_class="text-lg font-bold")
                Badge(f"{len(categories)} categories", variant="secondary")
            Separator()
            if categories:
                with Row(gap=4, css_class="items-start"):
                    with Column(gap=2, css_class="flex-1"):
                        Muted("Categories & Tools", css_class="text-xs uppercase tracking-wider")
                        for cat in categories:
                            tools = _cat_tool_list.get(cat, [])
                            with Card():
                                with CardContent(css_class="p-3"):
                                    with Row(gap=2, align="center"):
                                        Badge(cat, variant="default")
                                        Text(f"{len(tools)} tools")
                                    Text(", ".join(tools[:8]), css_class="text-xs text-muted mt-1")
                    with Column(gap=2, css_class="w-64"):
                        Muted("Tool Coverage", css_class="text-xs uppercase tracking-wider")
                        if cat_chart:
                            with Card():
                                with CardContent(css_class="p-3"):
                                    BarChart(
                                        data=cat_chart,
                                        series=[ChartSeries(data_key="tools", label="Tools")],
                                        x_axis="category",
                                        show_legend=False,
                                    )
            else:
                with Alert(variant="info"):
                    AlertTitle("No CTF data")
                    AlertDescription("No CTF categories loaded. Run a CTF challenge first.")
    return app


# ═════════════════════════════════════════════════════════════════════════════
# @app.ui() — Pentest Report
# ═════════════════════════════════════════════════════════════════════════════

@app.ui()
def pentest_report(target: str) -> PrefabApp:
    """Full pentest report — findings by severity with exploit chain."""
    surface = get_surface(target)
    findings = get_findings(target)
    plan_data = get_plan(target) if target else {}

    by_severity = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
    for f in findings:
        sev = f.get("severity", "info").lower()
        if sev in by_severity:
            by_severity[sev].append(f)
        else:
            by_severity.setdefault(sev, []).append(f)

    total = len(findings)
    crit_count = len(by_severity["critical"])
    high_count = len(by_severity["high"])
    ports = surface.get("ports", [])
    techs = surface.get("technologies", [])

    sev_data = [
        {"severity": s, "count": len(by_severity.get(s, []))}
        for s in ["critical", "high", "medium", "low", "info"]
    ]

    sev_variant = {"critical": "destructive", "high": "warning", "medium": "secondary", "low": "outline", "info": "outline"}

    with PrefabApp() as app:
        with Column(gap=4, css_class="p-4"):
            with Row(gap=3, align="center"):
                Icon(name="shield", size="default")
                Heading(f"Pentest Report", css_class="text-lg font-bold")
                Badge(f"{total} findings", variant="destructive" if crit_count > 0 else "secondary")
            with Row(gap=4, css_class="flex-wrap"):
                Metric(label="Target", value=target)
                Metric(label="Ports", value=str(len(ports)))
                Metric(label="Critical", value=str(crit_count))
                Metric(label="High", value=str(high_count))
            Separator()
            if findings:
                with Accordion(css_class="w-full"):
                    for sev, label in [("critical", "Critical"), ("high", "High"), ("medium", "Medium"), ("low", "Low"), ("info", "Info")]:
                        items = by_severity.get(sev, [])
                        if not items:
                            continue
                        with AccordionItem(title=f"{label} ({len(items)})"):
                            for f in items:
                                score = f.get("score", "")
                                score_str = f"Score: {score}" if score else ""
                                with Card():
                                    with CardContent(css_class="p-2"):
                                        with Column(gap=1):
                                            with Row(gap=2, align="center"):
                                                Badge(sev.upper(), variant=sev_variant.get(sev, "outline"))
                                                Text(f.get("finding", ""), css_class="font-mono text-sm")
                                                if score_str:
                                                    Muted(score_str, css_class="text-xs")
                                            if f.get("details"):
                                                Code(f["details"], css_class="text-xs")
                                            exploit = f.get("exploit", {})
                                            if exploit.get("tool"):
                                                with Row(gap=2, align="center"):
                                                    Muted("Exploit:", css_class="text-xs")
                                                    Badge(exploit["tool"], variant="warning")
                                                    Muted(f"confidence: {exploit.get('confidence', '')}", css_class="text-xs")
            else:
                with Alert(variant="info"):
                    AlertTitle("No findings")
                    AlertDescription(f"No vulnerabilities detected for {target}")
            if ports:
                Separator()
                Muted("Open Ports", css_class="text-xs uppercase tracking-wider")
                with Table():
                    TableCaption(content=f"{len(ports)} open ports on {target}")
                    with TableHeader():
                        with TableRow():
                            with TableHead(): Text("Port")
                            with TableHead(): Text("Service")
                            with TableHead(): Text("State")
                    with TableBody():
                        for p in ports:
                            with TableRow():
                                with TableCell(): Text(str(p.get("port", "")))
                                with TableCell(): Text(p.get("service", ""))
                                with TableCell(): Badge(p.get("state", ""), variant="outline")
    return app


# ═════════════════════════════════════════════════════════════════════════════
# @app.ui() — Recon Summary
# ═════════════════════════════════════════════════════════════════════════════

@app.ui()
def recon_summary(target: str) -> PrefabApp:
    """Reconnaissance summary — ports, tech, cache, history."""
    surface = get_surface(target)
    history = get_history(target, limit=20)
    cache_entries = _cache_for_target(target) if target else []

    ports = surface.get("ports", [])
    techs = surface.get("technologies", [])
    risk = surface.get("risk_level", "unknown")
    risk_var = surface.get("risk_variant", "default")

    risk_color = {"high": "destructive", "medium": "warning", "low": "success", "unknown": "secondary"}

    now = time.time()
    cache_info = [
        {"tool": e.get("tool", "?"), "age": _fmt_duration(now - e.get("timestamp", now)) if e.get("timestamp") else "?"}
        for e in cache_entries[-10:]
    ]

    tech_chart = [{"tech": t, "count": 1} for t in techs] if techs else [{"tech": "(none)", "count": 0}]

    with PrefabApp() as app:
        with Column(gap=4, css_class="p-4"):
            with Row(gap=3, align="center"):
                Icon(name="compass", size="default")
                Heading(f"Recon — {target}", css_class="text-lg font-bold")
                Badge(risk.upper(), variant=risk_color.get(risk, "outline"))
            with Row(gap=4, css_class="flex-wrap"):
                Metric(label="Open Ports", value=str(len(ports)))
                Metric(label="Technologies", value=str(len(techs)))
                Metric(label="Cache entries", value=str(len(cache_entries)))
                Metric(label="History entries", value=str(len(history)))
            Separator()
            if ports:
                Muted("Ports & Services", css_class="text-xs uppercase tracking-wider")
                with Table():
                    TableCaption(content="Port scan results")
                    with TableHeader():
                        with TableRow():
                            with TableHead(): Text("Port")
                            with TableHead(): Text("Service")
                            with TableHead(): Text("State")
                    with TableBody():
                        for p in ports:
                            with TableRow():
                                with TableCell(): Text(str(p.get("port", "")))
                                with TableCell(): Text(p.get("service", ""))
                                with TableCell(): Text(p.get("state", ""))
            if techs:
                Separator()
                Muted("Technology Stack", css_class="text-xs uppercase tracking-wider")
                with Card():
                    with CardContent(css_class="p-3"):
                        with Row(gap=2, css_class="flex-wrap"):
                            for t in techs:
                                Badge(t, variant="outline")
            if cache_info:
                Separator()
                Muted("Cache (recent)", css_class="text-xs uppercase tracking-wider")
                with DataTable(
                    columns=[
                        DataTableColumn(key="tool", header="Tool"),
                        DataTableColumn(key="age", header="Age"),
                    ],
                    rows=cache_info,
                ):
                    pass
            if history:
                Separator()
                Muted("Recent Tools", css_class="text-xs uppercase tracking-wider")
                with DataTable(
                    columns=[
                        DataTableColumn(key="tool", header="Tool"),
                        DataTableColumn(key="execution_display", header="Time"),
                        DataTableColumn(key="status", header="Status"),
                    ],
                    rows=history[-8:],
                ):
                    pass
    return app


# ═════════════════════════════════════════════════════════════════════════════
# @app.tool() — Pulse Dashboards Guide
# ═════════════════════════════════════════════════════════════════════════════

@app.tool(model=True)
def pulse_dashboards_guide(target: str = "") -> dict:
    """Discover and use Pulse UI dashboards — CTF tracker, pentest report, recon summary."""
    url = target.rstrip("/") if target else "<target>"
    return {
        "message": f"Call `search_tools(\"dashboard\")` to find available Pulse dashboards.",
        "dashboards": [
            {
                "name": "ctf_dashboard",
                "args": {},
                "description": "CTF challenge tracker — categories, tool coverage, BarChart",
                "search_hint": "dashboard ctf",
            },
            {
                "name": "pentest_report",
                "args": {"target": url},
                "description": "Findings by severity (Accordion), exploit Code blocks, port Table",
                "search_hint": "dashboard pentest",
            },
            {
                "name": "recon_summary",
                "args": {"target": url},
                "description": "Ports, tech Badges, DataTable cache + history",
                "search_hint": "dashboard recon",
            },
        ],
        "note": "Each returns a structured visual PrefabApp with icons, badges, data tables.",
    }


@app.tool()
def rev_entropy_map(file_path: str, block_size: int = 256, threshold: float = 7.0) -> dict:
    """Compute Shannon entropy per block of a binary file.
    
    Identifies packed, encrypted, or compressed regions — areas the LLM cannot
    analyze statically. High-entropy blocks (>=threshold) are flagged suspicious.
    Use this BEFORE calling rev_strings or other analysis to decide where to focus.
    
    Args:
        file_path: Path to the binary file
        block_size: Bytes per block (default 256)
        threshold: Entropy threshold for suspicious flag (default 7.0)
    
    Returns:
        dict with file metadata, block count, high_entropy_blocks, and
        per-block list of {block, offset, size, entropy, suspicious}
    """
    from mcp_core.rev_direct import compute_entropy_map
    return compute_entropy_map(file_path, block_size, threshold)


@app.tool()
def crypto_xor_crack(
    data: str = "",
    file_path: str = "",
    key_length_range: str = "1-20",
    top_n: int = 5,
    known_plaintext: str = "",
) -> dict:
    """Crack XOR-encrypted data using frequency analysis and known-plaintext derivation.
    
    Use when you have ciphertext (hex, base64, or raw) that might be XOR'd with
    a single-byte or repeating-key cipher. Tries all 256 single-byte keys, short
    repeating keys (2-6), and direct known-plaintext key derivation.
    
    Args:
        data: Hex, base64, or raw string input
        file_path: Load bytes from file instead of data string
        key_length_range: Range of key lengths e.g. "1-20" (default) or "3-3"
        top_n: Number of top candidates to return
        known_plaintext: If you know part of the plaintext (e.g. "flag{", "CTF"),
            provide it and the key is derived directly from cipher XOR known.
    
    Returns:
        dict with results sorted by score, each having key, key_bytes, plaintext,
        score, method. matched_known_plaintext if known_plaintext found in output.
    """
    from mcp_core.crypto_direct import xor_crack
    return xor_crack(
        data=data,
        file_path=file_path,
        key_length_range=key_length_range,
        top_n=top_n,
        known_plaintext=known_plaintext,
    )


@app.tool()
def crypto_z3_solve(
    constraints: str = "",
    file_path: str = "",
    variables: str = "",
    timeout_seconds: int = 30,
) -> dict:
    """Solve SMT constraints using Z3 (SMT-LIB2 syntax).
    
    Use for cryptographic puzzles, reverse engineering constraint solving, or
    any problem expressible as boolean formulas. Accepts SMT-LIB2 directly.
    Optionally declare BitVec(64) variables with the variables parameter.
    
    Args:
        constraints: SMT-LIB2 assertions (e.g. "(assert (= x #x41))")
        file_path: Read constraints from file instead
        variables: Comma-separated list of BitVec(64) variables to declare
        timeout_seconds: Solver timeout (default 30)
    
    Returns:
        dict with sat/unsat/unknown status and model (if sat).
        Error message if z3 is not installed.
    """
    from mcp_core.crypto_direct import z3_solve
    return z3_solve(
        constraints=constraints,
        file_path=file_path,
        variables=variables,
        timeout_seconds=timeout_seconds,
    )


@app.tool()
def crypto_rsa_attack(
    n: int,
    e: int,
    ciphertext: int = 0,
    attacks: str = "all",
    factordb: bool = False,
) -> dict:
    """Factor RSA modulus or break weak RSA keys.
    
    Use when you have RSA public key parameters (n, e) and need to recover
    the private key d. Tries small primes trial division, Fermat factorization,
    and Wiener's attack (small d). Optionally queries FactorDB API.
    
    Args:
        n: RSA modulus (integer)
        e: Public exponent (integer)
        ciphertext: Optional ciphertext to decrypt if factorization succeeds
        attacks: Comma-separated "small_primes,fermat,wiener,factordb" or "all"
        factordb: If True, query FactorDB API first (requires internet)
    
    Returns:
        dict with success, method, p/q/d if found, plaintext if ciphertext given.
    """
    from mcp_core.crypto_direct import rsa_attack
    return rsa_attack(
        n=n, e=e, ciphertext=ciphertext, attacks=attacks, factordb=factordb,
    )


@app.tool()
def binary_varint(
    data_hex: str = "",
    encoding: str = "leb128_unsigned",
) -> dict:
    """Decode a sequence of LEB128/varint values from hex-encoded binary data.

    Use on hex dumps from .sal files, protocol captures, or any binary blob
    containing variable-length integer sequences. Returns parsed integer values.

    Args:
        data_hex:  Hex string of varint-encoded bytes (e.g. "7201ff7f01")
        encoding:  "leb128_unsigned" (default), "leb128_signed", or "uleb128"

    Returns:
        dict with success, values (list of ints), count.
    """
    from mcp_core.binary_direct import binary_varint as _fn
    return _fn(data_hex=data_hex, encoding=encoding)


@app.tool()
def binary_block_parser(
    data_hex: str = "",
    block_size: int = 0,
    fields: str = "",
) -> dict:
    """Parse a hex-encoded binary blob into fixed-size blocks with typed fields.

    Use for reverse-engineering custom binary formats: .sal metadata blocks,
    file headers, protocol frames. Define fields as a JSON array.

    Args:
        data_hex:   Hex string of raw binary data
        block_size: Number of bytes per block
        fields:     JSON array of field dicts, each with name/offset/size/type.
                    Type: "uint8", "uint16", "uint32", "int32", "ascii", "bytes".
                    Example: '[{"name":"id","offset":0,"size":4,"type":"uint32"}]'

    Returns:
        dict with success, blocks (list of parsed fields per block), block_count.
    """
    import json as _json
    parsed_fields = []
    if fields:
        try:
            parsed_fields = _json.loads(fields)
        except _json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid fields JSON: {e}"}
    from mcp_core.binary_direct import binary_block_parser as _fn
    return _fn(data_hex=data_hex, block_size=block_size, fields=parsed_fields)


@app.tool()
def binary_rle_decode(
    data_hex: str = "",
    format: str = "value_count",
    initial_state: int = 0,
) -> dict:
    """Reconstruct a digital signal from RLE-encoded binary data (hex → varints → signal).

    Used for analyzing Saleae .sal files, signal captures, and any
    run-length encoded binary format. The hex data is parsed as varints,
    then interpreted as run lengths.

    Formats:
      "value_count":  [val, count, val, count, ...] — each pair explicit
      "toggle_count": [run1, run2, ...] — state flips each run (Saleae-style)
      "pulse_width":  [w1, w2, ...] — alternating high/low pulse widths

    Args:
        data_hex:      Hex string of varint-encoded run lengths
        format:        RLE variant ("value_count", "toggle_count", "pulse_width")
        initial_state: Starting state (0 or 1) for toggle_count / pulse_width

    Returns:
        dict with signal_values (truncated to 100k), sample_count, truncated flag.
    """
    from mcp_core.binary_direct import binary_rle_decode as _fn
    return _fn(data_hex=data_hex, format=format, initial_state=initial_state)


@app.tool()
def binary_signal_decode(
    data_hex: str = "",
    rle_values: str = "",
    protocol: str = "uart",
    baud: int = 9600,
    options: str = "",
) -> dict:
    """Decode a digital signal (UART/I2C/SPI) from samples or RLE run lengths.

    Primary use: decode UART 8N1 from Saleae .sal RLE data. Pass the varint
    values from binary_varint() as a JSON array in rle_values, or pass raw
    sample bytes as hex in data_hex.

    Args:
        data_hex:   Hex-encoded raw samples (each byte = 0 or 1)
        rle_values: JSON array of RLE run lengths (alt to data_hex)
        protocol:   "uart" (default, pure Python), "i2c" or "spi" (sigrok)
        baud:       Baud rate (default 9600, typical: 115200)
        options:    JSON dict with sample_rate (Hz, default 1000000),
                    rle_format, initial_state, invert

    Returns:
        dict with success, decoded text, bytes, frames, frame_count.
    """
    import json as _json
    parsed_opts = {}
    if options:
        try:
            parsed_opts = _json.loads(options)
        except _json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid options JSON: {e}"}
    from mcp_core.binary_direct import binary_signal_decode as _fn
    return _fn(
        data_hex=data_hex,
        rle_values=rle_values,
        protocol=protocol,
        baud=baud,
        options=parsed_opts,
    )


if __name__ == "__main__":
    app.run()
