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

import logging
import re
import time

from fastmcp import FastMCPApp

from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge, Card, CardContent,
    Column, DataTable, DataTableColumn, Div, ForEach,
    Grid, Heading, Icon, Metric, Muted,
    Progress, Row, Separator, Text, Tooltip,
)
from prefab_ui.components.charts import Sparkline
from prefab_ui.rx import Rx

from config import _config as app_config
from mcp_core.server_setup import _scan_cache, _rate_limit_events
from server_core.operational_metrics import _op_metrics
from server_core.singletons import enhanced_process_manager, error_handler, get_decision_engine, get_tool_stats_store
from tool_registry import TOOLS

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
SUCCESS_RATE    = Rx("success_rate_display").default("\u2014%")
CACHE_HIT_RATE  = Rx("cache_hit_display").default("\u2014%")
TIMEOUT_DISP    = Rx("timeout_count_display").default("0")
rx_runs      = TOTAL_RUNS_DISP
rx_success   = SUCCESS_RATE
rx_cache     = CACHE_HIT_RATE
rx_timeouts  = TIMEOUT_DISP


# ═════════════════════════════════════════════════════════════════════════════
# Backend tools
# ═════════════════════════════════════════════════════════════════════════════

@app.tool()
def get_overview() -> dict:
    """Header data: version, uptime, RAM, server status."""
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


@app.tool()
def get_surface(target: str | None = None) -> dict:
    """Return surface data (ports, services, techs, risk) for a target.

    Uses the active scope if target is None.
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
            if len(parts) >= 2 and "/" in parts[0]:
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

    return {
        "target":         target,
        "ports":          sorted(ports, key=lambda p: p["port"]),
        "ports_count":    port_count,
        "technologies":   sorted(techs),
        "risk_level":     risk,
        # Pre-formatted display
        "risk_variant":   "destructive" if risk == "high" else "warning" if risk == "medium" else "default",
        "ports_display":  f"{port_count} open port{'s' if port_count != 1 else ''}" if port_count else "No ports detected",
    }


@app.tool()
def get_findings(target: str | None = None) -> list[dict]:
    """Return vulnerabilities / issues for a target from nuclei and nikto.

    Uses the active scope if target is None.
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
        output = str(result.get("output", "") or result.get("stdout", ""))

        if tool == "nuclei":
            for line in output.splitlines():
                line = line.strip()
                m = re.match(
                    r"\[(critical|high|medium|low|info)\]\s*\[([^\]]*)\].*",
                    line, re.IGNORECASE,
                )
                if m:
                    sev = m.group(1).lower()
                    finding_id = m.group(2).strip()
                    parts = line.split()
                    url = parts[-1] if parts else ""
                    findings.append({
                        "tool": tool,
                        "severity": sev,
                        "finding": finding_id or url,
                        "details": url[:120],
                    })

        elif tool == "nikto":
            for line in output.splitlines():
                line = line.strip()
                if line.startswith("+ ") and "/:" in line:
                    findings.append({
                        "tool": tool,
                        "severity": "info",
                        "finding": line[2:].strip()[:100],
                        "details": "",
                    })

    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 5))
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


@app.tool()
def get_plan(target: str | None = None, objective: str = "comprehensive") -> dict:
    """Attack plan for a target via IntelligentDecisionEngine."""
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



# ═════════════════════════════════════════════════════════════════════════════
# UI entry point
# ═════════════════════════════════════════════════════════════════════════════

@app.ui()
def pulse_dashboard() -> PrefabApp:
    """Open the Pulse dashboard (3+2+1 grid layout)."""
    overview = get_overview()
    scope = get_scope()
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
    success_rate_display = f"{int(ops['global_success_rate'] * 100)}%" if ops['total_runs'] > 0 else "\u2014"
    cache_hit_display = f"{int(ops['cache']['hit_ratio'] * 100)}%" if ops['cache']['total'] > 0 else "\u2014"
    timeout_count_display = str(len(ops.get("timeout_count_by_tool", [])))

    err = get_errors_and_failures()
    err_sr = err.get("global_success_rate", 0)
    error_summary = (
        f"{err.get('total_errors', 0)} errors \u00b7 "
        f"{len(err.get('timeout_by_tool', []))} tools with timeouts \u00b7 "
        f"{int(err_sr * 100)}% success"
    )
    error_success_rate_display = f"{int(err_sr * 100)}%" if err.get("total_runs", 0) > 0 else "\u2014"

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
            Muted(f"{rx_success} success")
            Muted(f"\u26a0 {rx_timeouts} timeouts")

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
            "success_rate_display":  success_rate_display,
            "cache_hit_display":     cache_hit_display,
            "timeout_count_display": timeout_count_display,
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


if __name__ == "__main__":
    app.run()
