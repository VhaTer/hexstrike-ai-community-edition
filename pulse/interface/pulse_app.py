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
    python3 hexstrike.py mcp

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

from config import _config as app_config
from pulse.interface.server_setup import _scan_cache, _rate_limit_events, _optimizer, TOOL_TIMEOUTS
from pulse.tools.null_context import NullContext
from pulse.infrastructure.telemetry.operational_metrics import _op_metrics
from pulse.infrastructure.execution.process_manager import ProcessManager
from pulse.infrastructure.singletons import enhanced_process_manager, error_handler, get_decision_engine, get_target_store, get_tool_stats_store, get_ctf_manager
from pulse.intelligence.exploit_rules import suggest_exploit, ESTIMATED_TIMES, compute_layer2_score
from tool_registry import TOOLS
from pulse.tools.tool_registry_v2 import _registry
from pulse.interface.dashboard_sections import (
    SectionConfig,
    build_registry,
    detect_workflow_state as _ds_detect_workflow,
    auto_detect_sections,
    load_section as _ds_load_section,
    cost_for_sections,
    cache_for_target as _ds_cache_for_target,
)

logger = logging.getLogger(__name__)

app = FastMCPApp("Pulse Dashboard")

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
_TECH_KEYWORDS = {
    "wordpress": "WordPress", "joomla": "Joomla", "drupal": "Drupal",
    "nginx": "Nginx", "apache": "Apache", "php": "PHP",
    "python": "Python", "node.js": "Node.js", "java": "Java",
    "react": "React", "angular": "Angular", "vue": "Vue.js",
    "bootstrap": "Bootstrap", "jquery": "jQuery",
}
_WHATWEB_TECH_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.-]*)\[")
_WHATWEB_META_FIELDS = frozenset({
    "country", "ip", "meta-author", "script", "title", "httpserver",
})


# ═════════════════════════════════════════════════════════════════════════════
# UI entry points — passerelle (R5): the views live in
# pulse/interface/dashboards/, pulse_app wires them onto the app.
# ═════════════════════════════════════════════════════════════════════════════

from pulse.interface.dashboards import (
    ctf_dashboard as _dash_ctf_dashboard,
    pentest_report as _dash_pentest_report,
    pulse_dashboard as _dash_pulse_dashboard,
    recon_summary as _dash_recon_summary,
)
from pulse.interface.dashboards.pulse_dashboard import _STATE_KEY_SOURCES  # re-export (test contract)


@app.ui()
def pulse_dashboard() -> PrefabApp:
    """Open the Pulse dashboard (3+2+1 grid layout)."""
    return _dash_pulse_dashboard(_collect_dashboard_state())


@app.ui()
def ctf_dashboard() -> PrefabApp:
    """CTF challenge tracker — categories, tools, progress."""
    return _dash_ctf_dashboard()


@app.ui()
def pentest_report(target: str) -> PrefabApp:
    """Full pentest report — findings by severity with exploit chain."""
    return _dash_pentest_report(target)


@app.ui()
def recon_summary(target: str) -> PrefabApp:
    """Reconnaissance summary — ports, tech, cache, history."""
    return _dash_recon_summary(target)

# Backend tools
# ═════════════════════════════════════════════════════════════════════════════

@app.tool(model=True)
def get_overview() -> dict:
    """Discover the Pulse environment: version, uptime, RAM, disk, CPU, tools count, server health.

    Call FIRST when starting a fresh session. No target required — returns system-level info.
    Returns: version_display, uptime_display, ram_display (avail/total GB),
    cpu_percent, cpu_history (for sparklines), disk_percent, tools_count,
    server_status ('healthy' or 'limited'), total_runs, total_errors.

    Do NOT use for target-specific data — use get_scope() or get_live_dashboard(target) instead.
    Call BEFORE get_scope() to establish environment baseline.
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
        logger.debug("Failed to extract cpu_history from resource monitor", exc_info=True)

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
    """Detect current active target from recent scan cache or explicit input.

    Auto-discovers the most recently scanned target when called without args.
    Pass an explicit target to set scope without scanning.
    Returns: active_target, target_type (ip/domain/url), tools_used, tools_count,
    last_seen_ago, scope_summary.

    Call AFTER get_overview() to see what target Pulse is currently focused on.
    Call BEFORE get_surface() to ensure a target is selected.
    Do NOT use for scanning — use scan() or scan_background() instead.
    Example: get_scope()
    Example: get_scope('192.168.1.165')
    Next: get_surface() for port/tech data, or scan() to run recon tools
    """
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
    """Assess attack surface: open ports, services, and technology detection for a target.

    Parses cached nmap scan results for open ports/services + whatweb output for tech
    detection. Auto-uses active scope target if none provided.
    Risk levels: high (>5 open ports), medium (>2), low (>0), unknown.
    Also returns next_suggested_tool based on detected ports and technologies.

    Returns: target, risk_level, ports[] (port/service/state), port_count,
    technologies[], ports_display, risk_variant, next_suggested_tool{}.

    Call AFTER get_scope() or scan() to assess attack surface.
    Do NOT use for targets that haven't been scanned yet — run scan() first.
    Example: get_surface() — uses current scope target
    Example: get_surface('scanme.nmap.org')
    Next: get_findings() for vulnerability analysis
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
    app_name = ""
    for e in entries:
        if e.get("tool") != "whatweb":
            continue
        result = e.get("result", {})
        output = str(result.get("output", "") or result.get("stdout", "")).lower()
        for keyword, label in _TECH_KEYWORDS.items():
            if keyword in output:
                techs.add(label)
        # Secondary parser: extract any WhatWeb Name[value] tech names
        for m in _WHATWEB_TECH_RE.finditer(output):
            name = m.group(1)
            if name.lower() not in _WHATWEB_META_FIELDS and len(name) > 2:
                techs.add(name.capitalize())
        # Extract app name from Title field
        tm = re.search(r"title\[([^\]]+)\]", output)
        if tm:
            app_name = tm.group(1).strip()

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
        "app_name":       app_name,
        "risk_level":     risk,
        "risk_variant":   "destructive" if risk == "high" else "warning" if risk == "medium" else "default",
        "ports_display":  f"{port_count} open port{'s' if port_count != 1 else ''}" if port_count else "No ports detected",
    }
    if suggestion:
        result["next_suggested_tool"] = suggestion
    return result


@app.tool(model=True)
def get_findings(target: str | None = None) -> list[dict]:
    """Get vulnerabilities and security issues for a target from nuclei + nikto scan cache.

    Each finding is enriched with Couche 1 exploit suggestion (which tool to use)
    and Layer 2 score (0.0-1.0 based on severity, location, tool reliability, complexity).
    Returns findings sorted by Layer 2 score descending — highest priority first.

    Each finding: tool, severity, finding (ID or URL), details, exploit{tool, confidence},
    layer2{score, label, factors}, score (display string).

    Auto-uses active scope target if none provided.
    Call AFTER get_surface() to identify actual vulnerabilities on open ports.
    Do NOT use if no scan has been run — call scan(intensity='medium' or 'full') first.
    Example: get_findings()
    Example: get_findings('scanme.nmap.org')
    Next: get_plan() for attack chain, or follow next_suggested_tool from findings
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
    """Compare baseline vs live effectiveness for all tools with recorded runs.

    Returns per-tool: tool name, baseline effectiveness (0-1), live effectiveness,
    blended score, runs count, successes. Useful for understanding which tools
    perform best in your environment.

    No target required — shows global statistics across all sessions.
    Call for tool selection guidance — prefer tools with high blended scores.
    Example: get_tool_intelligence()
    """
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
    """Check rate limit detection state for a target.

    Shows current profile (aggressive/normal/conservative/stealth), confidence level,
    detection indicators, event history, and recommended timing parameters.
    Profiles adapt based on observed WAF/rate-limiting behavior.

    No target needed for global state. Pass target to filter by specific host.
    Call BEFORE running aggressive tools to avoid getting blocked.
    Example: get_rate_limit_status()
    Example: get_rate_limit_status('192.168.1.165')
    """
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
    """Get error classification, failure trends, and tool-specific error counts.

    Aggregates data from IntelligentErrorHandler (error types per tool, alternatives)
    and OperationalMetricsStore (error/timeout counts by tool, slowest tools).
    Useful for diagnosing tool failures and tuning timeouts.

    No target required — shows global error statistics across all sessions.
    Call when tools are failing unexpectedly to identify patterns.
    Returns: error_types[], error_counts_by_tool[], timeout_counts_by_tool[],
    slowest_tools[], summary.
    Example: get_errors_and_failures()
    """
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
        logger.debug("Failed to get error statistics", exc_info=True)

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
    """Generate an attack chain for a target from the IntelligentDecisionEngine.

    Produces ordered steps with tool name, expected outcome, success probability,
    and estimated execution time. Useful for planning exploitation after recon is complete.
    Auto-uses active scope target if none provided.

    Returns: target, steps[] (num, tool, expected_outcome, success_probability,
    execution_time_estimate, prob_display, eta_display, outcome_short),
    step_count, estimated_time (seconds), risk_level, summary.

    objective: comprehensive (default, full chain) | quick (fastest path) | stealth (low-noise).
    Falls back to empty steps if no target or IDE unavailable.

    Call AFTER get_findings() to plan exploitation based on discovered vulnerabilities.
    Do NOT use before recon — call get_surface() + get_findings() first.
    Example: get_plan()
    Example: get_plan('scanme.nmap.org', objective='stealth')
    Next: execute the first step(s) via scan() or run_security_tool()
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
    """Show what is actually running: live commands, background scans, resources.

    Data sources (all real):
    - active_processes: subprocesses currently running, from the process
      registry written by the command executor (one entry per live tool run)
    - active_workers: background scans (run_async_tool) still in progress
    - resource_usage: live CPU/memory/disk from the resource monitor

    No target required — shows system-wide process state.
    Call before launching new scans to check the machine is not overloaded,
    and after run_async_tool()/scan_background() to monitor progress.
    Example: get_active_tools()
    """
    try:
        registered = ProcessManager.list_active_processes()
        running = {
            pid: info
            for pid, info in registered.items()
            if info.get("status") == "running"
        }
        processes = [
            {
                "pid": pid,
                "command": info.get("command", ""),
                "status": info.get("status", ""),
                "progress": round(info.get("progress", 0.0), 2),
                "runtime": round(info.get("runtime", 0.0), 1),
                "eta": round(info.get("eta", 0.0), 1),
            }
            for pid, info in sorted(running.items())
        ]
        with _async_scans_lock:
            async_scans = [
                {
                    "scan_id": scan_id,
                    "tool": info.get("tool", ""),
                    "target": info.get("target", ""),
                    "status": info.get("status", ""),
                    "elapsed": round(time.time() - info.get("start_time", time.time()), 1),
                }
                for scan_id, info in _async_scans.items()
                if info.get("status") in ("starting", "running")
            ]
        resource = enhanced_process_manager.get_comprehensive_stats().get("resource_usage", {})
        return {
            "active_processes": len(processes),
            "active_workers": len(async_scans),
            "queue_size": 0,
            "processes": processes,
            "async_scans": async_scans,
            "resource_usage": resource,
            "summary": (
                f"{len(processes)} process(es) · {len(async_scans)} async scan(s)"
                f" · CPU {resource.get('cpu_percent', '?')}% · RAM {resource.get('memory_percent', '?')}%"
            ),
        }
    except Exception as e:
        logger.debug(f"get_active_tools unavailable: {e}")
        return {
            "active_processes": 0,
            "active_workers": 0,
            "queue_size": 0,
            "processes": [],
            "async_scans": [],
            "resource_usage": {},
            "summary": f"Unavailable: {str(e)[:80]}",
        }


@app.tool()
def get_history(target: str | None = None, limit: int = 50) -> list[dict]:
    """Get scan history from cache, optionally filtered by target.

    Returns recent scan entries sorted by timestamp (newest first).
    Each entry: tool, target, timestamp, age (human-readable), status, execution_time,
    execution_display, error.

    No target needed for all history. Pass target to filter by specific host.
    limit: max entries to return (default 50).
    Call BEFORE get_surface() or get_findings() to see what data is already cached.
    Example: get_history()
    Example: get_history('192.168.1.165', limit=10)
    """
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
    """Compare per-tool success rates, error counts, and timeout frequencies.

    Shows worst performers first (lowest success rate). Each entry: tool name,
    runs count, successes, errors, rate_display (percentage), timeouts.
    Also returns a separate timeouts list and overall summary.

    No target required — shows global statistics across all sessions.
    Call when diagnosing tool reliability issues or tuning timeouts.
    Example: get_tool_performance()
    """
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
    """Get scan cache hit/miss statistics and per-tool cache performance.

    Returns total hits, misses, hit_ratio, cache_size, max_size, utilization,
    hit_rate, and per-tool cache_hits breakdown. Useful for understanding
    how effectively the scan cache is serving repeated targets.

    No target required — shows global cache statistics.
    Call after repeated scans to verify cache is working as expected.
    Example: get_cache_status()
    """
    cs = _op_metrics.cache_summary()
    tool_hits = _op_metrics.cache_hits_by_tool()

    adv_stats = {}
    try:
        from pulse.infrastructure.singletons import cache
        adv_stats = cache.get_stats()
    except Exception:
        logger.debug("Failed to get cache stats", exc_info=True)

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
        from pulse.interface.server_setup import _scan_cache
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
    """Get CPU, memory, and disk usage trends over time.

    Returns cpu_avg (10-period), memory_avg, measurements count, period_minutes,
    cpu_history (30-point), mem_history (30-point), disk_display.
    Useful for monitoring system load during long-running scans.

    No target required — shows system-wide resource trends.
    Call when system feels slow to check if resources are constrained.
    Example: get_system_trends()
    """
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
    """Get active and completed scan session summaries.

    Returns counts of active vs completed sessions, with lists of recent sessions.
    Each completed session includes: session_id, target, findings count,
    tools_executed (list), timestamps, age_display.

    No target required — shows all sessions across all targets.
    Call to understand current workload and past session history.
    Example: get_sessions()
    """
    try:
        from pulse.infrastructure.singletons import get_session_store
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
    """Get user confirmation statistics for dangerous operations.

    Shows count of accepted, denied, and skipped confirmation prompts.
    Useful for understanding which operations users approve vs reject.

    No target required — global statistics across all sessions.
    Example: get_confirmations()
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
    """Get network I/O statistics — bytes sent and received since server start.

    Shows cumulative counters with human-readable formatting (B/KB/MB/GB).
    Useful for monitoring bandwidth usage during large scans or data transfers.

    No target required — system-wide network stats.
    Example: get_network_io()
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
# Section registry — dashboard_sections integration
# ═════════════════════════════════════════════════════════════════════════════


def _get_async_data(target: str | None = None) -> dict:
    """Return async scan data for the async section (no scan_id needed)."""
    with _async_scans_lock:
        running = [
            {"scan_id": sid, "tool": s["tool"], "target": s.get("target", "?"),
             "status": s["status"]}
            for sid, s in _async_scans.items()
            if s.get("status") in ("starting", "running")
        ][-10:]
        completed = [
            {"scan_id": sid, "tool": s["tool"], "target": s.get("target", "?"),
             "status": s["status"]}
            for sid, s in _async_scans.items()
            if s.get("status") in ("completed", "failed")
        ][-20:]
    return {
        "running": running,
        "completed": completed,
        "running_count": len(running),
        "completed_count": len(completed),
    }


def _has_async_scans() -> bool:
    """Check if any async scans exist."""
    with _async_scans_lock:
        return len(_async_scans) > 0


def _has_errors() -> bool:
    """Check if any errors have been recorded."""
    try:
        return _op_metrics.summary().get("total_errors", 0) > 0
    except Exception:
        return False


_SECTION_REGISTRY = build_registry([
    SectionConfig("header",       "HEADER",       get_overview,            cost_est=500,  always=True),
    SectionConfig("scope",        "SCOPE",        get_scope,               cost_est=300,  always=True),
    SectionConfig("surface",      "SURFACE",      get_surface,             cost_est=2000, requires_target=True, depends="scope"),
    SectionConfig("findings",     "FINDINGS",     get_findings,            cost_est=3000, requires_target=True, depends="findings"),
    SectionConfig("plan",         "PLAN IDE",     get_plan,                cost_est=1500, requires_target=True, depends="plan"),
    SectionConfig("history",      "HISTORY",      get_history,             cost_est=1500, condition=lambda: len(_scan_cache) > 0),
    SectionConfig("active",       "ACTIVE TOOLS", get_active_tools,        cost_est=300,  condition=_has_async_scans),
    SectionConfig("async",        "ASYNC SCANS",  _get_async_data,         cost_est=300,  condition=_has_async_scans),
    SectionConfig("errors",       "ERRORS",       get_errors_and_failures, cost_est=1000, condition=_has_errors),
    SectionConfig("performance",  "PERFORMANCE",  get_tool_performance,    cost_est=800,  condition=lambda: len(_scan_cache) > 5),
    SectionConfig("cache",        "CACHE",        get_cache_status,        cost_est=800,  condition=lambda: len(_scan_cache) > 0),
    SectionConfig("intel",        "INTEL",        get_tool_intelligence,   cost_est=500,  always=True),
    SectionConfig("trends",       "TRENDS",       get_system_trends,       cost_est=500,  always=True),
    SectionConfig("sessions",     "SESSIONS",     get_sessions,            cost_est=500),
    SectionConfig("confirmations","CONFIRMATIONS",get_confirmations,       cost_est=300),
    SectionConfig("netio",        "NETWORK I/O",  get_network_io,          cost_est=200),
])


# ═════════════════════════════════════════════════════════════════════════════
# Async scan execution — Phase 3 (fix timeout 300s)
# ═════════════════════════════════════════════════════════════════════════════

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
    Cancel it at any time with cancel_scan(scan_id).
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
            import asyncio as _asyncio
            from pulse.tools.null_context import NullContext
            from pulse.interface.server_setup import run_security_tool

            with _async_scans_lock:
                entry = _async_scans[scan_id]
                if entry.get("cancel_requested"):
                    entry.update({"status": "cancelled", "end_time": time.time()})
                    return
                entry["status"] = "running"

            start = time.time()
            null_ctx = NullContext()
            result = _asyncio.run(run_security_tool(null_ctx, tool, parsed_params))
            elapsed = time.time() - start

            with _async_scans_lock:
                entry = _async_scans[scan_id]
                entry["end_time"] = time.time()
                entry["status"] = "cancelled" if entry.get("cancel_requested") else "completed"
                entry["result"] = {
                    "success": result.get("success", False),
                    "stdout": (result.get("output", "") or "")[:2000],
                    "execution_time": elapsed,
                    "error": (result.get("error", "") or "")[:200],
                    "returncode": result.get("returncode", -1),
                }

        except Exception as e:
            with _async_scans_lock:
                entry = _async_scans[scan_id]
                entry["status"] = "cancelled" if entry.get("cancel_requested") else "failed"
                entry["end_time"] = time.time()
                entry["error"] = str(e)[:200]

    threading.Thread(target=_run, daemon=True, name=f"async-{tool}").start()
    _cleanup_old_scans()

    return {"scan_id": scan_id, "status": "started", "tool": tool, "target": target}


@app.tool()
def get_scan_status(scan_id: str) -> dict:
    """Poll the status of an async scan launched via run_async_tool().

    Returns current state: running (with elapsed time + ETA), cancelled,
    completed (with result), failed (with error message), or not_found.

    Args:
        scan_id: The scan_id returned by run_async_tool()

    Returns:
        dict with scan_id, status (starting/running/cancelled/completed/failed/not_found),
        tool, target, elapsed (seconds), eta_seconds + eta_display (when running),
        result (if completed/cancelled), error (if failed)

    Each call is idempotent and lightweight (~0ms). Poll every 2-5 seconds
    for running scans. Returns 'not_found' for invalid or expired scan_ids.

    Example: get_scan_status('scan_abc123')
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

    if entry["status"] in ("starting", "running"):
        avg = _op_metrics.avg_duration_by_tool().get(entry["tool"], 0.0)
        result["eta_seconds"] = round(max(avg - elapsed, 0.0), 1) if avg else None
        result["eta_display"] = ESTIMATED_TIMES.get(entry["tool"], "1-10 min")

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


@app.tool()
def cancel_scan(scan_id: str) -> dict:
    """Cancel a background scan launched via run_async_tool().

    Marks the task as cancelled immediately (polling sees it right away) and
    best-effort terminates the running command. To avoid killing a parallel
    scan of the same tool, the command is only terminated when this is the
    only running scan for that tool; otherwise only the task status changes.

    Args:
        scan_id: The scan_id returned by run_async_tool()

    Returns:
        dict with scan_id, status (cancelled/not_found), already_final (when
        the scan had already finished), terminated_pids (commands killed).
    """
    with _async_scans_lock:
        entry = _async_scans.get(scan_id)
        if entry is None:
            return {"scan_id": scan_id, "status": "not_found"}
        current = entry["status"]
        if current in ("completed", "failed", "cancelled"):
            return {"scan_id": scan_id, "status": current, "already_final": True}
        tool = entry.get("tool", "")
        entry["cancel_requested"] = True
        entry["status"] = "cancelled"
        same_tool_running = sum(
            1 for s in _async_scans.values()
            if s.get("tool") == tool and s.get("status") in ("starting", "running")
        )

    terminated: list[int] = []
    if same_tool_running <= 1:
        for pid, info in ProcessManager.list_active_processes().items():
            command = str(info.get("command", ""))
            if info.get("status") == "running" and command.startswith(tool + " "):
                if ProcessManager.terminate_process(pid):
                    terminated.append(pid)
                    logger.info(f"🛑 cancel_scan({scan_id}): terminated pid {pid} ({command[:60]})")

    return {
        "scan_id": scan_id,
        "status": "cancelled",
        "terminated_pids": terminated,
        "note": "scan marked cancelled — poll get_scan_status(scan_id) for the final state",
    }


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


def _suggest_next_from_context(surface: dict, findings: list, exclude: set = frozenset()) -> dict:
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
            if techs_lower:
                return {"tool": "gobuster", "reason": f"Tech identified ({', '.join(techs_lower[:3])}) — discover hidden paths", "expected_time": "1-5 min", "priority": "high"}
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
        logger.debug("Failed to record scan in TargetStore for %s", active_target, exc_info=True)

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
    """Full Pulse dashboard state in one call — replaces 15+ individual get_* calls.

    Returns all 18 panels in a single response: Overview, Scope, Surface, Findings,
    Plan, Active Tools, History, Rate Limit, Errors & Failures, Tool Performance,
    Cache Status, System Trends, Sessions, Confirmations, Network I/O, Async Scans,
    Intelligence, next_suggested_tool. ~100ms typical response.

    Each panel contains pre-formatted display strings and raw data for agent processing.

    Call this INSTEAD of calling individual get_* tools when you need a full picture.
    No target needed for system-wide stats. Pass target to filter scope/surface/findings
    to a specific host.

    Do NOT use for single-tool operations — use scan() or individual tools instead.
    Do NOT use before any scan has run — panels will be empty.

    Example: get_live_dashboard() — all targets
    Example: get_live_dashboard('192.168.1.165') — filtered to one target
    Next: check next_suggested_tool in response, then follow the recommendation
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
# Section-based dashboard — lightweight, auto-detected sections
# ═════════════════════════════════════════════════════════════════════════════


@app.tool(model=True)
def get_dashboard(
    sections: str | list[str] | None = None,
    target: str | None = None,
) -> dict:
    """Pulse dashboard with auto-detected sections — lightweight alternative to get_live_dashboard().

    Uses session state to determine which panels are relevant, then loads only
    those panels. Token cost scales with number of sections loaded (~300-3000
    tokens per section).

    sections: comma-separated or list of section names. If None, auto-detect.
    target: optional target to filter scope/surface/findings.

    Available sections: header, scope, surface, findings, plan, history, active,
    async, errors, performance, cache, intel, trends, sessions, confirmations, netio.

    Example: get_dashboard() → auto-detect
    Example: get_dashboard("header,scope,surface") → explicit sections
    Example: get_dashboard(["header", "scope"], "192.168.1.165") → filtered
    """
    _registry_ref = _SECTION_REGISTRY

    if sections is None:
        selected = auto_detect_sections(
            _registry_ref, _scan_cache, get_scope,
            has_async_scans_fn=_has_async_scans,
            has_failures_fn=_has_errors,
        )
    elif isinstance(sections, str):
        selected = [s.strip() for s in sections.split(",") if s.strip() in _registry_ref]
    else:
        selected = [s for s in sections if s in _registry_ref]

    if not selected:
        selected = ["header", "scope"]

    cost_info = cost_for_sections(selected, _registry_ref)

    section_data = {}
    for name in selected:
        section_data[name] = _ds_load_section(name, _registry_ref, target)

    return {
        "sections": selected,
        "section_count": len(selected),
        "total_cost_est": cost_info["total_cost"],
        "data": section_data,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3.5 — Pulse workflow guide (agent-agnostic entry point)
# ═════════════════════════════════════════════════════════════════════════════

_WORKFLOW_STEPS = {
    "overview": {
        "label": "Environment Overview",
        "description": "Establish baseline — check Pulse version, resources, and server health",
        "tools": ["get_overview()"],
    },
    "scope": {
        "label": "Target Selection",
        "description": "Select or detect active target for scanning",
        "tools": ["get_scope()", "scan(target, intensity='quick')", "scan_background(target, intensity='medium')"],
    },
    "surface": {
        "label": "Surface Analysis",
        "description": "Discover open ports, services, and technologies",
        "tools": ["get_surface()", "get_history()", "get_live_dashboard()"],
    },
    "findings": {
        "label": "Vulnerability Detection",
        "description": "Find vulnerabilities through nuclei/nikto scan results",
        "tools": ["get_findings()", "get_live_dashboard()"],
    },
    "plan": {
        "label": "Attack Planning",
        "description": "Generate attack chain with success probabilities and ETAs",
        "tools": ["get_plan()", "get_plan(objective='stealth')"],
    },
    "exploit": {
        "label": "Exploitation",
        "description": "Execute attack: suggested tools, CTF solving, or manual primitives",
        "tools": ["ctf_analyze()", "ctf_solve()", "execute_code()", "http_request()", "run_security_tool()", "run_async_tool()"],
    },
}

_WORKFLOW_ORDER = ["overview", "scope", "surface", "findings", "plan", "exploit"]


def _detect_workflow_state() -> tuple[str | None, str, dict]:
    """Detect current workflow state (delegates to dashboard_sections)."""
    return _ds_detect_workflow(_scan_cache, get_scope)


@app.tool(model=True)
def pulse_guide(step: str | None = None) -> dict:
    """Guide the agent through the Pulse workflow — tells you what to do next.

    Detects current session state from scan cache and returns the recommended
    next step. Call without arguments for full workflow status with your current
    position. Pass a step name (overview|scope|surface|findings|plan|exploit)
    to drill into available tools for that phase.

    Workflow: overview → scope → surface → findings → plan → exploit

    Returns: workflow[], current_step, next_step, tools_available[],
    context{active_target, tools_run, tools_count, reason}.

    Call FIRST when connecting to Pulse — tells you exactly where to start.
    Do NOT use during an active scan — use get_scan_status() or get_active_tools() instead.

    Example: pulse_guide()
    Example: pulse_guide('surface')  — tools for surface analysis
    Example: pulse_guide('exploit')  — exploitation primitives
    """
    if step:
        step = step.lower()
        if step in _WORKFLOW_STEPS:
            info = _WORKFLOW_STEPS[step]
            idx = _WORKFLOW_ORDER.index(step)
            return {
                "step": step,
                "label": info["label"],
                "description": info["description"],
                "tools_available": info["tools"],
                "previous_step": _WORKFLOW_ORDER[idx - 1] if idx > 0 else None,
                "next_step": _WORKFLOW_ORDER[idx + 1] if idx < len(_WORKFLOW_ORDER) - 1 else None,
            }
        return {"error": f"Unknown step '{step}'. Valid steps: {', '.join(_WORKFLOW_ORDER)}"}

    current, nxt, ctx = _detect_workflow_state()
    workflow = []
    for i, s in enumerate(_WORKFLOW_ORDER):
        info = _WORKFLOW_STEPS[s]
        workflow.append({
            "step": s,
            "label": info["label"],
            "active": s == current or (current is None and s == "overview"),
            "completed": _WORKFLOW_ORDER.index(s) < _WORKFLOW_ORDER.index(current) if current else False,
            "tools": info["tools"],
        })

    return {
        "workflow": workflow,
        "current_step": current,
        "next_step": nxt,
        "summary": f"Current: {current or 'not started'} → Next: {nxt}",
        "context": ctx,
        "quick_start": {
            "no_target": "scan('target-ip')",
            "with_target": "scan() → get_surface() → get_findings() → get_plan()",
            "ctf": "ctf_analyze() → scan(target) → ctf_solve()",
            "pwn": "execute_code(pwntools) for binary exploitation",
        },
        "exploit_hint": ctx.get("reason", ""),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Unified scan entry point
# ═════════════════════════════════════════════════════════════════════════════

@app.tool(model=True)
def scan(target: str = "", intensity: str = "quick", objective: str = "comprehensive") -> dict:
    """Run reconnaissance scan on a target — the primary entry point for target analysis.

    Executes security tools based on intensity level, returns surface analysis,
    vulnerability findings, attack plan, and next_suggested_tool in one response.
    Uses scan cache — recently scanned targets return instantly.

    Intensity levels:
    - quick (default): nmap + whatweb — open ports and tech detection. ~30s uncached.
    - medium: + nuclei + nikto — adds vulnerability scanning. ~2-3 min.
    - full: + gobuster (web targets) — complete recon with directory busting. ~5-10 min.

    Returns: target, intensity, tools{} (per-tool status/duration/cached),
    surface{} (ports, technologies, risk_level), findings[] (enriched with exploit+layer2),
    plan{} (attack chain steps), next_suggested_tool{}, summary, cache_age.

    objective: comprehensive (default) | quick | stealth — guides the attack planner.
    target: IP, URL, or hostname. Auto-detects from scope if left empty.

    Call FIRST for any new target — replaces separate nmap/whatweb/nuclei calls.
    Do NOT use for background tasks >30s — use scan_background() instead (returns task_id).
    Do NOT use for single-tool debugging — use run_security_tool() directly.

    Example: scan('192.168.1.165')
    Example: scan('http://example.com', intensity='full')
    Next: get_findings() for detailed vulnerability analysis, or follow next_suggested_tool
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
    tool_results = {}

    _workers = min(len(tools_to_run), 5)
    with ThreadPoolExecutor(max_workers=_workers) as pool:
        futures = {pool.submit(_run_scan_tool, name, resolved): name
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
        logger.debug("Failed to record scan in TargetStore for %s", resolved, exc_info=True)

    suggestion = _suggest_next_from_context(surface_data, findings_data)

    # Don't re-suggest a tool that just ran
    if suggestion and suggestion.get("tool") in tools_to_run:
        completed_names = {t for t, d in tool_results.items() if d.get("status") in ("completed", "cached")}
        if not findings_data and intensity == "quick":
            suggestion = {"tool": "scan", "reason": "Quick scan complete — run medium intensity for vulnerability detection", "expected_time": "2-5 min", "priority": "high"}
        elif not findings_data:
            suggestion = {"tool": "http_request", "reason": "Web app detected — probe endpoints manually", "expected_time": "1-5 min", "priority": "medium"}
        else:
            suggestion = {"tool": "get_findings", "reason": "Review findings before next step", "expected_time": "0s", "priority": "medium"}

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
        "next_suggested_tool": suggestion or None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# UI entry point
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cache_for_target(target: str) -> list[dict]:
    """Return all scan cache entries for a given target (delegates to dashboard_sections)."""
    return _ds_cache_for_target(_scan_cache, target)


def _run_scan_tool(tool_name: str, resolved: str) -> dict:
    """Run a single scan tool via run_security_tool. Thread-safe — runs in ThreadPoolExecutor.

    Returns {tool_name, status, error?, returncode?}.
    Cache check is done first — cache hits return immediately without
    calling run_security_tool().
    """
    import asyncio
    from pulse.interface.server_setup import run_security_tool

    cache_key = f"sess:{tool_name}:{resolved}"
    try:
        if cache_key in _scan_cache:
            cached = _scan_cache[cache_key]
            cached_stdout = cached.get("result", {}).get("stdout", "")
            # Invalidate nmap cache entries with "0 hosts up"
            if tool_name in _TOOLS_NEED_HOST and cached_stdout.lstrip().startswith("Starting"):
                if "0 hosts up" in cached_stdout or "0 IP addresses" in cached_stdout:
                    pass  # fall through to re-execute
                else:
                    return {"tool_name": tool_name, "status": "cached", "cached": True,
                            "returncode": cached.get("result", {}).get("returncode")}
            else:
                return {"tool_name": tool_name, "status": "cached", "cached": True,
                        "returncode": cached.get("result", {}).get("returncode")}
    except Exception:
        pass

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
            parsed = urlparse(resolved)
            host = parsed.hostname or resolved
            port = parsed.port
            params = {"target": host, "scan_type": "-sTV"}
            if port:
                params["ports"] = str(port)

        params = _optimizer.optimize(tool_name, params)

        null_ctx = NullContext()
        result = asyncio.run(run_security_tool(null_ctx, tool_name, params))

        ok = result.get("success", False)
        if ok:
            _scan_cache[cache_key] = {
                "tool": tool_name,
                "target": resolved,
                "timestamp": time.time(),
                "result": {"stdout": result.get("stdout", ""),
                           "output": result.get("output", ""),
                           "success": True},
            }
            return {"tool_name": tool_name, "status": "completed",
                    "returncode": result.get("returncode")}
        timed_out = result.get("timed_out", False)
        error = result.get("error", "")
        if timed_out:
            return {"tool_name": tool_name, "status": "timeout", "error": error}
        return {"tool_name": tool_name, "status": "failed", "error": error}
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


if __name__ == "__main__":
    app.run()
