# Phase 1 Freeze Direction Audit

Date: 2026-05-01  
Scope: compare `hexstrike-ai_pulse_stabilisation_plan.Md` Phase 1 against current code and recent commits.

## Phase 1 Requirement

Phase 1 asks the project to freeze direction before deeper stabilization work.

Required output:

- active entrypoint
- supported MCP tool exposure
- supported workflow prompts
- supported dashboard scope
- supported compatibility surface, if any

Exit criteria:

- agreement on what is officially supported during stabilization
- no ambiguous ownership between standalone runtime and transitional compatibility code

## Current Git Facts

Current HEAD observed:

```text
cae0d9a feat(observability): Phase 4 complete - OperationalMetricsStore + metrics://tools resource
```

Recent commits after `4cd532b`:

| Commit | Summary | Phase impact |
|---|---|---|
| `b1a8436` | `plan_attack` restores IDE profile from session state | Phase 2/3 behavior |
| `a2ec6b4` | normalize result shape, fix `plan_attack` chain, fix cache API | Phase 2 stabilization |
| `a55da80` | populate telemetry: cache hit, session state, timeout, target | Phase 4 telemetry |
| `cae0d9a` | add `OperationalMetricsStore` and `metrics://tools` | Phase 4 observability |

Git cannot prove which human or assistant authored the work; commits are authored as `VhaTer`. If these commits were produced by Claude, the concrete work is the table above.

Current working tree also shows tracked root docs deleted:

```text
D COVERAGE_ANALYSIS.md
D EXECUTIVE_SUMMARY.md
D LOW_COVERAGE_ANALYSIS_REPORT.md
D README.md
D README_TESTS.md
D hexstrike-fastmcp-knowloadge-base.md
```

That is documentation hygiene debt, not a runtime Phase 1 implementation.

## Supported Surface: Code Facts

### Active Entrypoint

Verdict: defined in code.

The active runtime entrypoint is `hexstrike_server.py`, which builds the standalone FastMCP server and runs HTTP transport.

Evidence:

- `hexstrike_server.py` registers dashboard/health routes.
- `hexstrike_server.py` calls `mcp.run(transport="http", port=API_PORT, show_banner=False)`.
- `mcp_core/server_setup.py` exposes `setup_mcp_server_standalone(...)` as the no-Flask runtime path.

### Supported MCP Tool Exposure

Verdict: mostly defined in code.

Supported MCP exposure currently includes:

- generic `run_security_tool(...)`
- generated typed wrappers for direct routes that resolve through the tool registry
- `get_tool_skill(...)`
- `plan_attack(...)`
- MCP resources, including `metrics://tools`

Direct route facts:

- `DIRECT_TOOLS` contains 106 direct routes.
- 105 typed wrappers were previously verified from direct routes with registry definitions/aliases.
- 16 `mcp_core/*_direct.py` modules exist.

Risk:

- The supported surface is implicit in code, not declared in one canonical project document.

### Supported Workflow Prompts

Verdict: defined in code.

Implemented prompts:

- `bug_bounty_recon`
- `wifi_attack_chain`
- `ctf_web_challenge`
- `smb_lateral_movement`
- `cloud_security_audit`

These prompts are FastMCP-native and instruct clients to call `run_security_tool(...)`.

### Supported Dashboard Scope

Verdict: partially defined.

FastMCP custom routes in `hexstrike_server.py` expose:

- `/dashboard`
- `/health`
- `/ping`
- `/web-dashboard`
- `/web-dashboard/stream`
- static-file catch-all when `server_static/` exists

Risk:

- Legacy Flask-style dashboard modules still exist under `server_api/`.
- `pytest.ini` excludes `tests/test_endpoints_exist.py`, explicitly marking Flask route tests obsolete.
- The supported dashboard surface should be documented as FastMCP custom routes only, unless legacy Flask routes are intentionally supported.

### Supported Compatibility Surface

Verdict: not fully closed.

Evidence of transitional compatibility still present:

- `setup_mcp_server(...)` remains in `mcp_core/server_setup.py` and is documented as "Phase 2 - Flask still present".
- `server_api/` contains Flask blueprint route modules.
- `mcp_core/hexstrike_client.py` still references reaching a Flask server.
- `pytest.ini` excludes Flask endpoint tests from the default run.

Interpretation:

- The active runtime path is clear: FastMCP standalone.
- The compatibility path is present but not clearly declared as supported, unsupported, or legacy-only.
- This means Phase 1 exit criteria are not fully met yet.

## What Was Done After The Previous Audit

The recent code changes are useful but mostly belong to later phases:

### Done: Result Shape Normalization

`server_core/command_executor.py` now normalizes executor output into canonical keys:

- `success`
- `output`
- `error`
- `returncode`
- `timed_out`
- `partial_results`
- `execution_time`
- `timestamp`

It also preserves legacy aliases:

- `stdout`
- `stderr`
- `return_code`

This supports Phase 2 runtime consistency, not Phase 1 freeze definition.

### Done: Cache API Fix

`execute_command(...)` now uses `AdvancedCache.get(command)` and `AdvancedCache.set(command, result)` consistently.

This is a runtime stabilization fix.

### Done: `plan_attack` Session Restore

`plan_attack(...)` now attempts to restore `ide_profile:{target}` from session state using `TargetProfile.from_dict(...)` before re-analysis.

This supports Phase 3 workflow/intelligence behavior.

### Done: Tool Telemetry

`run_security_tool(...)` telemetry now tracks:

- target
- timed out state
- cache hit
- session state restore
- confirmation state
- optimizer profile
- skill injection
- prompt suggestion

This supports Phase 4 observability.

### Done: Operational Metrics Resource

`server_core/operational_metrics.py` adds an in-memory metrics store.

`mcp_core/server_setup.py` exposes:

```text
metrics://tools
```

This supports Phase 4 observability.

## Phase 1 Verdict

| Phase 1 item | Current status | Verdict |
|---|---|---|
| Freeze architectural churn | Recent commits are focused, but Phase 4 work started before Phase 1 was fully documented | Partial |
| Active entrypoint defined | `hexstrike_server.py` + `setup_mcp_server_standalone(...)` | Done in code |
| Supported MCP tool exposure defined | `run_security_tool`, typed wrappers, `get_tool_skill`, `plan_attack`, resources | Mostly done in code |
| Supported workflow prompts defined | 5 FastMCP prompts | Done in code |
| Dashboard scope defined | FastMCP custom routes exist; legacy dashboard modules remain | Partial |
| Compatibility surface defined | Legacy code exists but support status is not explicit | Not done |
| No ambiguous ownership | FastMCP path is active, but legacy Flask code remains ambiguous | Partial / not fully met |

## Phase 1 Blocking Gaps

1. No single canonical "supported surface" document exists yet.
2. Legacy `server_api/` and compatibility setup code remain present without explicit support status.
3. Dashboard support is split between active FastMCP custom routes and legacy Flask modules.
4. Root docs are deleted in the working tree while replacement docs live in ignored `Projects_reports_docs/`.

## Recommended Phase 1 Actions

1. Add a short supported-surface document for stabilization.
2. Explicitly mark legacy Flask/API routes as unsupported or compatibility-only during stabilization.
3. State that FastMCP custom dashboard routes are the supported dashboard surface.
4. Decide whether `Projects_reports_docs/` should be tracked or remain private/local.
5. Only after that, continue Phase 2 runtime stabilization work.

## Proposed Supported Surface

For the stabilization window:

- Active entrypoint: `hexstrike_server.py`.
- MCP runtime setup: `mcp_core/server_setup.py::setup_mcp_server_standalone`.
- Generic tool execution: `run_security_tool(...)`.
- Typed MCP tools: generated wrappers for direct routes with registry definitions.
- Workflow prompts: the 5 prompts in `mcp_core/prompts.py`.
- Resources: `health://server`, `scan://{target}/latest`, `scan://{target}/{tool_name}`, `scan://cache/list`, `metrics://tools`.
- Dashboard: FastMCP custom routes in `hexstrike_server.py`.
- Legacy Flask/API modules: compatibility code only; not part of the default stabilization quality gate unless explicitly re-enabled.
