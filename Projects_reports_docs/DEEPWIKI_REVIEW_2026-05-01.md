# DeepWiki Report — Review and Scope Classification

Date: 2026-05-01
Report source: Projects_reports_docs/deepwiki-report.md
Reviewed by: Claude (Anthropic)
Context: Pre-stable-candidate review during Phase 5

---

## Summary

DeepWiki identified 5 structural tensions in the codebase. All 5 are real and
accurately described. All 5 are in the legacy Flask architecture that is explicitly
out of scope for the stabilization window (see SUPPORTED_SURFACE.md).

None of them affect the active FastMCP standalone path.

---

## Tension-by-Tension Classification

### 1. Double path to the same Flask endpoint

Typed tools (`httpx_probe()`) and gateway (`run_tool("httpx")`) both hit
`POST /api/tools/httpx` with different validation contracts.

**Scope:** `setup_mcp_server()` (Flask-era) + `mcp_tools/gateway.py`
**Active path affected:** No — `setup_mcp_server_standalone()` has no Flask routes.
Typed wrappers call `run_security_tool()` directly, not `safe_post()`.
**Action:** None during stabilization. Document as legacy debt.

---

### 2. Dead cache on /api/command

`execute_command(command, use_cache=True)` without passing the cache instance
silently bypasses the cache. The Flask `/api/command` route never passes the
cache instance.

**Scope:** `hexstrike_server.py::generic_command()` Flask route
**Active path affected:** No — `run_security_tool()` uses `execute_command()`
via `*_direct.py` modules which use the module-level `_cache` singleton from
`server_core/singletons.py`. The cache is never dead on the standalone path.
**Action:** None during stabilization. Document as legacy debt.

---

### 3. Two unsynchronized taxonomies

`tool_registry.py` has 8 broad categories. `mcp_core/tool_profiles.py` has 20+
fine-grained profiles. No cross-reference mechanism.

**Scope:** `setup_mcp_server()` uses `tool_profiles.py` to load tool groups.
`setup_mcp_server_standalone()` uses `DIRECT_TOOLS` dict — a flat explicit map
with no dependency on either taxonomy.
**Active path affected:** No.
**Action:** None during stabilization. The divergence is harmless on standalone path.

---

### 4. TelemetryCollector instantiated twice

`enhanced_command_executor.py` creates `telemetry = TelemetryCollector()` (instance A).
`hexstrike_server.py` exposes a different `telemetry.get_stats()` on `/health` (instance B).
Stats may not reflect real executions.

**Scope:** Flask server `/health` endpoint
**Active path affected:** No — the standalone path uses `OperationalMetricsStore`
(`_op_metrics`) via `metrics://tools` resource. `TelemetryCollector` is not used
anywhere in `setup_mcp_server_standalone()`.
**Action:** None during stabilization.

---

### 5. Triple-layer validation without shared contract

FastMCP type-checking + gateway manual loop + Flask handler validation.
No shared Pydantic/JSON Schema source of truth.

**Scope:** Typed tool signatures + `mcp_tools/gateway.py` + Flask tool handlers
**Active path affected:** No — standalone path has two layers only:
1. FastMCP type-checking on typed wrappers
2. `run_security_tool()` which validates `isinstance(params, dict)` and routes
   through `_normalize_tool_result()`.
No Flask handler in the loop.
**Action:** None during stabilization.

---

## What DeepWiki Did Not See

DeepWiki analyzed the codebase at a snapshot that included legacy Flask code. It
did not distinguish between `setup_mcp_server()` (legacy) and
`setup_mcp_server_standalone()` (active). The standalone path has none of these
tensions because:

- No Flask routes — no dual-path problem
- Cache passed via module-level singleton — no dead cache
- `DIRECT_TOOLS` flat map — no taxonomy conflict
- `OperationalMetricsStore` is a single in-process singleton — no double instance
- Two-layer validation only — no triple-layer problem

---

## Legacy Debt Register (post-stabilization backlog)

These items are real and should be addressed when the legacy path is either
retired or explicitly re-stabilized:

| Item | Files | Priority |
|---|---|---|
| Unify typed/gateway validation contract | `mcp_tools/gateway.py`, `mcp_tools/web_probe/*.py` | P2 post-stable |
| Fix dead cache on `/api/command` | `hexstrike_server.py` | P2 post-stable |
| Single `TelemetryCollector` instance | `enhanced_command_executor.py`, `hexstrike_server.py` | P2 post-stable |
| Sync tool_registry ↔ tool_profiles taxonomies | `tool_registry.py`, `mcp_core/tool_profiles.py` | P3 post-stable |
| Retire or explicitly mark `setup_mcp_server()` | `mcp_core/server_setup.py` | P2 post-stable |

---

## Phase 5 Impact

Zero. DeepWiki findings do not affect the stable candidate criteria:
- Active path tests: unaffected
- `run_security_tool()` integration: unaffected
- Cache/session paths: unaffected
- Destructive confirmation: unaffected
- Normalization pipeline: unaffected

Phase 5 regression validation proceeds as planned.
