# HexStrike AI-PULSE — Agent Guide

## Setup

```bash
source hexstrike-env/bin/activate
python3 hexstrike_mcp.py --debug          # stdio MCP server
python3 hexstrike_server.py               # HTTP server
fastmcp dev apps pulse_app.py             # Prefab UI validation
```

## Test commands

```bash
python -m pytest tests/ --ignore=tests/test_real_integration.py --ignore=tests/test_server_setup_standalone.py -n auto -q
python -m pytest tests/test_pulse_app.py -v -q --tb=short
```

## Key conventions

- **Branch**: `feature/prefab-dashboard` — single branch for all Prefab dashboard work
- **Merge**: only when user judges stable. No auto-merge.
- **Prefab UI**: `@app.tool()` for backend, `@app.ui()` for entry point. State set once at creation.
- **Scope**: active target auto-detected from most recent `_scan_cache` entry.
- **Data sources**: `_op_metrics` for operational stats, `_scan_cache` for scan results, `get_tool_stats_store()` for tool effectiveness.
- **State keys**: flat dict in `PrefabApp(state={...})`, accessed via `Rx("key").then(lambda v: ..., default)`.

## Session 27 — 2026-05-14 (cleanup + Prefab 6 panels)

- Removed 5 orphan deps (selenium, aiohttp, mitmproxy, beautifulsoup4, webdriver-manager)
- Full Prefab dashboard with 6 panels (Status Bar, Resource Gauges, Cache+Errors, Execution Activity, Tools by Category, Intelligence)
- 4 backend tools + 1 UI entry point
- Branch `feature/prefab-dashboard` merged to master, tagged v0.8.0
- 2524 passed, 1 skipped, 2 warnings

## Session 28 — 2026-05-14 (v0.8.0 Prefab v2 — Header + Scope)

- New workflow: feature branch -> test -> merge when stable
- `_op_metrics.system_metrics()` now returns `memory_total_gb`
- `get_overview()` tool: version, uptime, RAM, tools count, server status
- `get_scope()` tool: auto-detects active target from most recent `_scan_cache` entry, detects type (ip/domain/url), shows tools used + timing
- **Header panel**: one-liner `PULSE v0.8.0 • up Xh Ym • RAM X/Y GB • N tools • ✓ healthy`
- **Scope panel**: Card with target, type badge, tool badges, summary line
- Legacy panels kept: System Resources, Recent Activity, Intelligence DataTable
- 5 dev panels removed: Cache+Errors, Tools by Category, old get_pulse_metrics
- 26 tests, 2550 total, 0 regressions

## Session 29 — 2026-05-15 (v0.8.0 Prefab v2 — Full dashboard with Plan IDE + Active Tools + History)

- Surface + Findings panels (from v0.8.0 Session B plan)
- `get_surface(target)` tool: parses nmap stdout for open ports/services, whatweb for tech detection, computes risk level (high >5 ports, medium >2 ports). Falls back to scope.
- `get_findings(target)` tool: parses nuclei terminal output (`[severity] [id] ...`) sorted by severity. Parses nikto (`+ /: finding`) for info issues.
- `_cache_for_target(target)` helper: filters `_scan_cache` by target across all sessions.
- `get_plan(target, objective)` tool: attack chain from `IntelligentDecisionEngine`, returns steps with prob/ETA.
- `get_active_tools()` tool: running processes from `EnhancedProcessManager`.
- `get_history(target, limit)` tool: scope-filtered scan history, replaces `get_recent_scans()`.
- `get_pulse_data(target)` tool: aggregates all 10 tools into a single JSON response.
- **Plan IDE panel**: DataTable with #/Tool/Outcome/Prob/ETA.
- **Active Tools panel**: Card with Metrics (Processes/Workers/Queued) + summary.
- **Historique panel**: DataTable (Tool/Target/When/Status/Time) replaces Recent Activity.
- Binary name fixes (7 tools underscore→hyphen), Pulse CLI removed, CTF made effective, orphaned files cleaned.
- 53 tests, 2580 total, 0 regressions.

## Session 30 — 2026-05-15 (Lazy init fixes — ParameterOptimizer, CTF automator, os.makedirs, pre-warming)

- `intelligent_decision_engine.py`: `ParameterOptimizer()` moved from module-level (eager) to `_get_parameter_optimizer()` lazy method on class. Saves ~50ms on first `get_decision_engine()` call.
- `ctf/automator.py`: removed module-level `CTFWorkflowManager()` and `CTFToolManager()` instances (duplicate singletons). Replaced with `self._manager` / `self._tools` lazy properties that delegate to `get_ctf_manager()` / `get_ctf_tools()` from singletons.
- `config_core.py`: added `resolve_data_dir()` + `ensure_data_dir()` (thread-safe, idempotent). `tool_stats_store.py`, `session_store.py`, `wordlist_store.py` now use it — eliminates 2 redundant `os.makedirs()` syscalls.
- `mcp_entry.py`: added background `_prewarm_singletons()` thread that pre-initializes `get_decision_engine()` + `get_tool_stats_store()` after server start. First dashboard call no longer blocks ~100-350ms.
- 2580 passed, 1 skipped, 2 warnings — 0 regressions.
