"""Pulse main dashboard UI — extracted from pulse_app.py (R5 passerelle).

Pure prefab_ui rendering: receives the collected dashboard state and
returns a PrefabApp. Data gathering stays in pulse.interface.pulse_app.
"""

from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Badge, Card, CardContent, Column,
    DataTable, DataTableColumn, Div, ForEach,
    Grid, Icon, Metric, Muted, Progress,
    Row, Separator, Tab, Tabs, Text, Tooltip,
)
from prefab_ui.components.charts import ChartSeries, LineChart, PieChart, Sparkline
from prefab_ui.rx import Rx

from pulse.interface.dashboards._helpers import _fmt_rl_summary, _rl_variant, _NST_VARIANTS

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
rx_rl_var = Rx("rl_variant").default("default")

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


# Footer
TOTAL_RUNS_DISP = Rx("total_runs_display").default("0 runs")
rx_runs      = TOTAL_RUNS_DISP

# Badge variant for next suggested tool priority (display)
rx_nst_tool   = Rx("nst_tool").default("No suggestion")
rx_nst_reason = Rx("nst_reason").default("Run a scan to get a recommendation")
rx_nst_time   = Rx("nst_time").default("")
rx_nst_var    = Rx("nst_variant").default("default")

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
TREND_SERIES   = Rx("trend_series").default([])
rx_trend_cpu  = TREND_CPU_AVG
rx_trend_mem  = TREND_MEM_AVG
rx_trend_per  = TREND_PERIOD
rx_trend_meas = TREND_MEASURES
rx_trend_cpu_hist = TREND_CPU_HIST
rx_trend_mem_hist = TREND_MEM_HIST
rx_trend_disk = TREND_DISK
rx_trend_series = TREND_SERIES

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

# Findings severity breakdown (PieChart)
FINDINGS_SEV = Rx("findings_by_severity").default([])
rx_findings_sev = FINDINGS_SEV

_STATE_KEY_SOURCES: dict[str, str] = {
    # ── Overview (st["overview"]) ──────────────────────────────────────────
    "version":               "overview['version']",
    "version_display":       "overview['version_display']",
    "uptime_display":        "overview['uptime_display']",
    "ram_display":           "overview['ram_display']",
    "tools_display":         "overview['tools_display']",
    "uptime_seconds":        "overview['uptime_seconds']",
    "ram_percent":           "overview['ram_percent']",
    "ram_available_gb":      "overview['ram_available_gb']",
    "ram_total_gb":          "overview['ram_total_gb']",
    "disk_percent":          "overview['disk_percent']",
    "cpu_percent":           "overview['cpu_percent']",
    "cpu_history":           "overview['cpu_history']",
    "cpu_display":           "overview['cpu_display']",
    "ram_detail_display":    "overview['ram_detail_display']",
    "disk_display":          "overview['disk_display']",
    "server_status":         "overview['server_status']",
    "server_status_variant": "overview['server_status_variant']",
    "tools_count":           "overview['tools_count']",
    "total_runs":            "overview['total_runs']",
    "total_errors":          "overview['total_errors']",
    # ── Footer stats ────────────────────────────────────────────────────────
    "total_runs_display":    "st['total_runs_display']",
    # ── Scope ───────────────────────────────────────────────────────────────
    "scope_target":          "scope['active_target']",
    "scope_type":            "scope['target_type']",
    "scope_tools":           "scope['tools_used']",
    "scope_tools_count":     "scope['tools_count']",
    "scope_last_seen_ago":   "scope['last_seen_ago']",
    "scope_age":             "scope['age_seconds']",
    "scope_summary":         "scope['scope_summary']",
    # ── Surface ─────────────────────────────────────────────────────────────
    "surface_target":        "surface['target']",
    "risk_level":            "surface['risk_level']",
    "risk_variant":          "surface['risk_variant']",
    "ports_display":         "surface['ports_display']",
    "ports_count":           "surface['ports_count']",
    "surface_ports":         "surface['ports']",
    "surface_techs":         "surface['technologies']",
    # ── Findings / system ───────────────────────────────────────────────────
    "findings":              "st['findings']",
    "findings_by_severity":  "derived: findings count by severity",
    "system":                "st['sys']",
    # ── Plan IDE ────────────────────────────────────────────────────────────
    "plan_target":           "plan['target']",
    "plan_steps":            "plan['steps']",
    "plan_summary":          "plan['summary']",
    # ── Active Tools ────────────────────────────────────────────────────────
    "active_processes":      "active['active_processes']",
    "active_workers":        "active['active_workers']",
    "active_queue":          "active['queue_size']",
    "active_summary":        "active['summary']",
    # ── History ─────────────────────────────────────────────────────────────
    "history":               "st['history']",
    # ── Rate Limit ──────────────────────────────────────────────────────────
    "rl_profile":            "rl['profile']",
    "rl_variant":            "_rl_variant(rl['profile'])",
    "rl_confidence":         "rl['confidence']",
    "rl_delay":              "rl['timing']['delay']",
    "rl_threads":            "rl['timing']['threads']",
    "rl_timeout":            "rl['timing']['timeout']",
    "rl_summary":            "_fmt_rl_summary(rl)",
    "rl_confidence_display": "derived: confidence → %",
    "rl_delay_display":      "derived: delay → ms",
    "rl_events":             "st['rl_events_table']",
    # ── Errors & Failures ───────────────────────────────────────────────────
    "error_total":               "err['total_errors']",
    "error_success_rate_display": "st['error_success_rate_display']",
    "error_summary":             "st['error_summary']",
    "error_by_tool":             "err['error_by_tool']",
    "timeout_by_tool":           "err['timeout_by_tool']",
    "slowest_tools":             "err['slowest_tools']",
    "error_by_type":             "err['error_by_type']",
    "recent_errors":             "err['recent_errors']",
    # ── Tool Performance ────────────────────────────────────────────────────
    "tool_performance":   "perf['tools']",
    "perf_timeouts":      "perf['timeouts']",
    "perf_summary":       "perf['summary']",
    # ── Missing Tools ───────────────────────────────────────────────────────
    "missing_tools":  "st['missing_tools']",
    "missing_count":  "derived: len(st['missing_tools'])",
    # ── Cache Status ────────────────────────────────────────────────────────
    "cache_hits":              "cache_status['hits']",
    "cache_misses":            "cache_status['misses']",
    "cache_hit_ratio_display": "st['cache_hit_ratio_display']",
    "cache_size":              "cache_status['cache_size']",
    "cache_max_size":          "cache_status['max_size']",
    "cache_util_display":      "st['cache_util_display']",
    "cache_summary_text":      "st['cache_summary_text']",
    "cache_by_tool":           "cache_status['by_tool']",
    # ── Cache Intelligence ──────────────────────────────────────────────────
    "cache_ttl_scores":  "st['cache_ttl_scores']",
    "cache_ttl_summary": "st['cache_ttl_summary']",
    # ── System Trends ───────────────────────────────────────────────────────
    "trend_cpu_avg_display": "st['trend_cpu_avg_display']",
    "trend_mem_avg_display": "st['trend_mem_avg_display']",
    "trend_period_display":  "st['trend_period_display']",
    "trend_measurements":    "trends['measurements']",
    "trend_cpu_history":     "trends['cpu_history']",
    "trend_mem_history":     "trends['mem_history']",
    "trend_disk_display":    "trends['disk_display']",
    "trend_series":          "derived: zip(cpu/mem history)",
    # ── Sessions ────────────────────────────────────────────────────────────
    "sessions_active":    "sessions['active']",
    "sessions_completed": "sessions['completed']",
    "sessions_summary":   "sessions['summary']",
    # ── Confirmations ───────────────────────────────────────────────────────
    "conf_accepted": "confirmations['accepted']",
    "conf_denied":   "confirmations['denied']",
    "conf_skipped":  "confirmations['skipped']",
    "conf_summary":  "confirmations['summary']",
    # ── Network I/O ─────────────────────────────────────────────────────────
    "net_sent_display":  "netio['bytes_sent_display']",
    "net_recv_display":  "netio['bytes_recv_display']",
    "net_total_display": "netio['total_display']",
    # ── Async scans ─────────────────────────────────────────────────────────
    "async_scans_running":  "st['async_scans_running']",
    "async_scans_complete": "st['async_scans_complete']",
    "async_scans_summary":  "st['async_scans_summary']",
    # ── Intelligence ────────────────────────────────────────────────────────
    "intelligence":          "get_tool_intelligence()",
    # ── Next suggested tool ─────────────────────────────────────────────────
    "next_suggested_tool":   "st['next_suggested_tool']",
    "nst_tool":              "derived: nst['tool']",
    "nst_reason":            "derived: nst['reason']",
    "nst_time":              "derived: nst['expected_time']",
    "nst_variant":           "derived: _NST_VARIANTS[nst['priority']]",
}

def _build_ui_state(st: dict) -> dict:
    """Flatten _collect_dashboard_state() into the PrefabApp state dict (R3).

    Single source of truth for the UI state keys. Every key must stay in
    sync with _STATE_KEY_SOURCES below — the parity test
    (test_state_keys_match_documented_contract) fails on any drift.
    """
    overview = st["overview"]
    scope = st["scope"]
    surface = st["surface"]
    findings = st["findings"]
    plan = st["plan"]
    active = st["active"]
    history = st["history"]
    rl = st["rl"]
    rl_events_table = st["rl_events_table"]
    sys = st["sys"]
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
    nst = st.get("next_suggested_tool", {}) or {}

    cpu_hist = trends.get("cpu_history", []) or []
    mem_hist = trends.get("mem_history", []) or []
    n_points = min(len(cpu_hist), len(mem_hist))
    trend_series = [
        {"idx": i, "cpu": cpu_hist[i], "mem": mem_hist[i]}
        for i in range(n_points)
    ]

    _SEV_ORDER = ["critical", "high", "medium", "low", "info"]
    sev_counts: dict[str, int] = {}
    for f in findings:
        if isinstance(f, dict):
            sev = str(f.get("severity") or "info").lower()
        else:
            sev = "info"
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    findings_by_severity = [
        {"severity": sev.title(), "count": count}
        for sev, count in sorted(
            sev_counts.items(),
            key=lambda kv: (_SEV_ORDER.index(kv[0]) if kv[0] in _SEV_ORDER
                            else len(_SEV_ORDER), kv[0]),
        )
    ]

    return {
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
        "findings_by_severity":  findings_by_severity,
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
        "trend_series":          trend_series,
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
        "intelligence":          _get_intelligence(),
        # Next suggested tool
        "next_suggested_tool":   st.get("next_suggested_tool", {}),
        "nst_tool":              nst.get("tool", ""),
        "nst_reason":            nst.get("reason", ""),
        "nst_time":              nst.get("expected_time", ""),
        "nst_variant":           _NST_VARIANTS.get(nst.get("priority", ""), "default"),
    }




def _get_intelligence():
    from pulse.interface.pulse_app import get_tool_intelligence  # lazy (avoid import cycle)
    return get_tool_intelligence()


def pulse_dashboard(st: dict) -> PrefabApp:
    """Open the Pulse dashboard — 3-zone layout (S92 redesign).

    Overview & Workflow (default view) / Findings / History. Header and
    Scope stay as a global bar above the tabs; the NEXT TOOL bandeau and
    footer stay global below them.
    """
    overview = st["overview"]

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

        # ── 3 zones: Overview & Workflow | Findings | History ──────────
        with Tabs(value="overview", variant="line", css_class="px-2 pt-1"):

            # ── Zone 1 — Overview & Workflow ────────────────────────────
            with Tab("Overview & Workflow", value="overview"):

                # Grid 3 colonnes: Surface | Plan IDE | Active Tools
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

                # ── System Trends (LineChart CPU/MEM) ──────────────────
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
                    LineChart(
                        data=Rx("trend_series"),
                        series=[
                            ChartSeries(data_key="cpu", label="CPU"),
                            ChartSeries(data_key="mem", label="MEM"),
                        ],
                        x_axis="idx",
                        height=140,
                        show_legend=True,
                    )

                # ── Cache Status ────────────────────────────────────────
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

                # ── Cache Intelligence ──────────────────────────────────
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

                # ── Missing Tools ───────────────────────────────────────
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

                # ── Rate Limit ──────────────────────────────────────────
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

                # ── Intelligence ────────────────────────────────────────
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

                # ── Network I/O ─────────────────────────────────────────
                Muted("NETWORK I/O", css_class="text-xs uppercase tracking-wider p-4")
                with Column(gap=2, css_class="px-4 pb-4"):
                    with Card():
                        with CardContent(css_class="p-3"):
                            with Row(gap=4):
                                Metric(label="Sent",     value=Rx("net_sent_display"))
                                Metric(label="Received", value=Rx("net_recv_display"))
                                Metric(label="Total",    value=Rx("net_total_display"))

            # ── Zone 2 — Findings ───────────────────────────────────────
            with Tab("Findings", value="findings"):

                Muted("FINDINGS BY SEVERITY", css_class="text-xs uppercase tracking-wider p-4")
                with Row(gap=4, css_class="px-4 pb-2 items-start flex-wrap"):
                    with Card():
                        with CardContent(css_class="p-3"):
                            PieChart(
                                data=Rx("findings_by_severity"),
                                data_key="count",
                                name_key="severity",
                                height=180,
                                inner_radius=50,
                                show_legend=True,
                            )

                Muted("DETAILS", css_class="text-xs uppercase tracking-wider px-4 pt-2")
                with Column(gap=2, css_class="px-4 pb-4"):
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

            # ── Zone 3 — History ────────────────────────────────────────
            with Tab("History", value="history"):

                # ── History (scans récents) ─────────────────────────────
                Muted("HISTORY", css_class="text-xs uppercase tracking-wider p-4")
                with Column(gap=2, css_class="px-4 pb-4"):
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

                # ── Errors & Failures ───────────────────────────────────
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

                # ── Tool Performance ─────────────────────────────────────
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

                # ── Sessions ─────────────────────────────────────────────
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

                # ── Confirmations ────────────────────────────────────────
                Muted("CONFIRMATIONS", css_class="text-xs uppercase tracking-wider p-4")
                with Column(gap=2, css_class="px-4 pb-4"):
                    with Card():
                        with CardContent(css_class="p-3"):
                            with Row(gap=4):
                                Metric(label="Accepted", value=Rx("conf_accepted"))
                                Metric(label="Denied",   value=Rx("conf_denied"))
                                Metric(label="Skipped",  value=Rx("conf_skipped"))
                    Muted(f"{rx_conf_sum}", css_class="text-sm text-muted pt-1")

        # ── Next suggested tool ─────────────────────────────────────────
        with Row(gap=3, align="center", css_class="p-2 px-4 bg-muted/10 border-b flex-wrap"):
            Muted("NEXT TOOL", css_class="text-xs uppercase tracking-wider")
            Badge(f"{rx_nst_tool}", variant=rx_nst_var)
            Muted(f"{rx_nst_reason}", css_class="text-sm flex-1 min-w-0 truncate")
            Muted(f"{rx_nst_time}", css_class="text-xs whitespace-nowrap")

        Separator()

        # ── Footer ─────────────────────────────────────────────────────
        with Row(gap=4, align="center", css_class="p-2 px-4 bg-muted/20 border-t flex-wrap"):
            Muted(f"{rx_version}")
            Muted(f"{rx_runs}")

    return PrefabApp(
        view=view,
        state=_build_ui_state(st),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
