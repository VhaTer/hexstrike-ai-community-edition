# Phase 2 Runtime Stabilization Audit

Date: 2026-05-01  
Scope: compare `hexstrike-ai_pulse_stabilisation_plan.Md` Phase 2 against the current active runtime code.

Fix pass: 2026-05-01  
Owner handoff: Phase 2 bad-flow fixes were applied after the initial audit.
Validation: targeted Phase 2/FastMCP tests passed (`96 passed in 285.19s`); full configured suite passed (`1323 passed, 6 xfailed, 2 warnings in 79.56s`).

## Phase 2 Requirement

Goal: make the real Phase 3 execution path predictable and consistent.

Primary focus:

- `hexstrike_server.py`
- `mcp_core/server_setup.py`
- `mcp_core/*_direct.py`
- active typed MCP tool calls
- `run_security_tool(...)`

Questions to close:

- do typed tools and `run_security_tool(...)` behave consistently?
- are confirmations enforced correctly on the standalone path?
- are cache and context interactions deterministic?
- are errors and timeouts normalized?
- are supported tools exposed coherently across the active surface?

Concrete work items:

- standardize result and error shapes on the active path
- verify timeout behavior and failure propagation
- validate tool registration coverage against the intended active surface
- remove or document remaining route drift between standalone and compatibility layers
- verify state restoration and cache reads are safe and non-surprising

## What Was Done Recently

Recent commits after `4cd532b` added runtime stabilization and observability work:

| Area | Code evidence | Phase 2 relevance |
|---|---|---|
| Result normalization | `server_core/command_executor.py::_normalize_result(...)` | Directly relevant |
| Cache API fix | `execute_command(...)` now calls `cache.get(command)` / `cache.set(command, result)` | Directly relevant |
| Timeout canonical fields | normalized result includes `timed_out`, `partial_results`, `error` | Directly relevant |
| `plan_attack` session restore | `TargetProfile.from_dict(...)` + `ctx.get_state("ide_profile:...")` | Phase 2/3 boundary |
| Runtime telemetry | `run_security_tool(...)` adds target/cache/session/timeout telemetry | Phase 2/4 boundary |
| Operational metrics | `metrics://tools` resource | Phase 4, but depends on Phase 2 telemetry |
| Cache API compatibility | `execute_command(...)` supports both `AdvancedCache` and `HexStrikeCache` signatures | Directly relevant |

## Question-By-Question Verdict

### 1. Typed tools and `run_security_tool(...)` consistency

Verdict: mostly consistent.

Evidence:

- `_create_typed_tool_wrapper(...)` builds typed wrappers around `run_security_tool(...)`.
- Typed wrappers collect required and optional parameters into a payload.
- Typed wrappers call:

```python
return await run_security_tool(ctx, tool_name, payload)
```

Runtime implication:

- Typed tools and generic calls share the same execution path after wrapper parameter collection.
- This is the correct architecture for Phase 2.

Remaining risk:

- `run_security_tool(...)` documents `parameters: str`, but typed wrappers pass a dict. The function supports both string and dict, but the type annotation/docstring is narrower than runtime behavior.
- Early returns from `run_security_tool(...)` do not use the full canonical result shape.

Status: mostly done, minor documentation/type-shape cleanup needed.

### 2. Destructive confirmations on standalone path

Verdict: enforced for the main destructive tools.

Evidence:

- `_DESTRUCTIVE_TOOLS` includes `aireplay_ng`, `mdk4`, `responder`, `metasploit`, `mitm6`.
- `_build_destructive_confirmation(...)` implements exceptions:
  - `aireplay_ng` attack mode `9` does not require confirmation.
  - `responder` analyze mode does not require confirmation.
  - Metasploit `auxiliary/scanner/*` and `auxiliary/gather/*` do not require confirmation.
- `confirm_destructive_action(...)` blocks execution if elicitation is unsupported, rejected, cancelled, or fails.
- `tests/test_server_setup_standalone.py` covers representative confirmation and skip paths.

Remaining risk:

- Denied confirmation returns before `_op_metrics.record(...)`, so `metrics://tools` will not count denied confirmations from this path.
- Invalid JSON and unknown tool returns also bypass metrics recording.

Status: safety enforcement done; observability for early exits incomplete.

### 3. Cache and context interactions

Verdict: partially deterministic, but one behavior needs review.

What works:

- Tech profile lookup order is deterministic:
  1. explicit `_tech` parameter
  2. `ctx.get_state(f"tech:{target}")`
  3. `_detect_from_cache(target)`
  4. best-effort `ctx.set_state(...)`
- Rate-limit profile restoration uses `ctx.get_state(f"ratelimit:{target}")`.
- Successful results are written to `_scan_cache`.
- Prior successful `_scan_cache` entries can short-circuit execution.

Risk:

- The cache-hit short-circuit returns before `_op_metrics.record(...)`, so metrics do not count cache-hit tool invocations.
- Cache lookup key is `f"{tool_name}:{target}"`. If equivalent typed/generic calls use different aliases or target normalization, they may miss the same cached result.
- Early cache hits return prior result exactly as stored. If older cache entries have older shapes, the path does not normalize them before returning.

Status: functional but not fully non-surprising.

### 4. Errors and timeouts normalized

Verdict: improved, not complete.

Done:

- `EnhancedCommandExecutor` marks timeout results with:
  - `success: False`
  - `timed_out: True`
  - `partial_results`
  - `return_code: -1`
- `server_core/command_executor.py::_normalize_result(...)` maps executor output to canonical keys:
  - `success`
  - `output`
  - `error`
  - `returncode`
  - `timed_out`
  - `partial_results`
  - `execution_time`
  - `timestamp`
- It preserves legacy aliases:
  - `stdout`
  - `stderr`
  - `return_code`
- `tests/test_command_executor_normalize.py` covers success, stderr fallback, failure, timeout, canonical keys, and cache storage.

Remaining gaps:

- `run_security_tool(...)` early returns for invalid JSON, unknown tool, and cancellation only return `success` and `error`.
- Some direct modules may still return minimal shapes such as `{"success": False, "error": ...}`.
- Cache-hit results are not re-normalized before return.

Status: command execution normalized; full active-path result normalization still partial.

### 5. Supported tools exposed coherently

Verdict: mostly coherent for FastMCP standalone; compatibility drift remains.

Current active surface:

- `run_security_tool(...)`
- generated typed MCP wrappers
- `get_tool_skill(...)`
- `plan_attack(...)`
- resources:
  - `health://server`
  - `scan://{target}/latest`
  - `scan://{target}/{tool_name}`
  - `scan://cache/list`
  - `metrics://tools`

Known route facts from previous verification:

- 106 direct routes.
- 105 typed wrappers.
- 16 direct modules.

Remaining drift:

- `setup_mcp_server(...)` still exists as a transitional setup path.
- `server_api/` Flask-style route modules still exist.
- `pytest.ini` excludes legacy Flask endpoint tests.
- `mcp_core/hexstrike_client.py` still references reaching a Flask server.

Status: active FastMCP surface coherent; compatibility surface still needs explicit documentation from Phase 1.

## Runtime Bugs / Risks Found During Phase 2 Audit

### P1: Metrics not recorded for early returns

Status after fix pass: fixed in `mcp_core/server_setup.py`.

Affected paths:

- invalid JSON
- unknown tool
- destructive action denied
- cache-hit short-circuit

Why it matters:

- Phase 4 metrics depend on Phase 2 telemetry.
- Denied confirmations and cache hits are key operational signals.
- `metrics://tools` may underreport real calls.

Applied fix:

- Added a local `finalize(result)` helper in `run_security_tool(...)`.
- `finalize(...)` normalizes result shape, sets telemetry success/timeout/duration, logs telemetry, records `_op_metrics`, and returns the normalized result.
- Invalid JSON, non-object JSON, unknown tools, destructive denial, cache hit, and normal execution now route through finalization.

### P2: Canonical result shape is not universal

Status after fix pass: mostly fixed for `run_security_tool(...)` return paths.

Affected paths:

- `run_security_tool(...)` early errors
- destructive cancellation
- direct module minimal error returns
- cache-hit returns if stored result has older/minimal shape

Why it matters:

- Callers cannot rely on `output`, `returncode`, `timed_out`, and `partial_results` being present.

Applied fix:

- Added `_normalize_tool_result(...)` in `mcp_core/server_setup.py`.
- It preserves existing result keys while ensuring canonical keys exist:
  - `success`
  - `output`
  - `error`
  - `returncode`
  - `timed_out`
  - `partial_results`
  - `execution_time`
  - `timestamp`
- Cache-hit results are normalized before return.

### P2: Parameter type contract is narrower than implementation

Status after fix pass: fixed.

`run_security_tool(...)` annotates `parameters: str`, but typed wrappers pass a dict and the implementation accepts both.

Applied fix:

- Updated `run_security_tool(...)` annotation to `parameters: str | Dict[str, Any]`.
- Updated docstring to state JSON string or dict.
- Added validation for parsed JSON values that are not objects.

### P2: Compatibility ownership still unclear

The active path is FastMCP standalone, but legacy Flask-compatible modules remain.

Likely fix:

- Close Phase 1 first by documenting support status.
- Do not stabilize legacy routes unless they are explicitly in scope.

### P1: `execute_command(...)` cache API drift

Status after fix pass: fixed in `server_core/command_executor.py`.

Issue found during validation:

- `execute_command(...)` used `cache.get(command)` and `cache.set(command, result)`.
- The shared singleton cache is `HexStrikeCache`, which requires `get(command, params)` and `set(command, params, result)`.
- Real `run_security_tool(...)` calls through direct modules could fail before executing the external command.

Applied fix:

- Added `_cache_get(...)` and `_cache_set(...)`.
- These support both `AdvancedCache` and `HexStrikeCache`.
- Added a regression test with real `HexStrikeCache`.

## Test Coverage Observed

Relevant tests exist:

- `tests/test_command_executor_normalize.py`
- `tests/test_server_setup_standalone.py`
- `tests/test_fastmcp3_ctx_methods.py`
- `tests/test_operational_metrics.py`
- `tests/test_skills_modernization.py`
- `tests/test_fastmcp_context_regressions.py`

Validation run:

```text
96 passed in 285.19s
```

Gaps to add:

- `run_security_tool(...)` invalid JSON records metrics. Added in `tests/test_server_setup_standalone.py`.
- non-object JSON records metrics. Added in `tests/test_server_setup_standalone.py`.
- unknown tool records metrics. Added in `tests/test_server_setup_standalone.py`.
- destructive denial records metrics. Added in `tests/test_server_setup_standalone.py`.
- cache-hit path records metrics. Added in `tests/test_server_setup_standalone.py`.
- cache-hit result is normalized before return. Added in `tests/test_server_setup_standalone.py`.
- typed wrapper and generic call produce the same normalized output for the same mocked direct route.

## Phase 2 Verdict

| Phase 2 item | Current status | Verdict |
|---|---|---|
| Typed tools and generic tool share execution path | Typed wrappers call `run_security_tool(...)` | Done |
| Destructive confirmations enforced | Main safety gates and exceptions implemented/tested | Done |
| Result normalization | Executor output normalized, early returns still minimal | Partial |
| Timeout behavior | Executor timeout cannot be success; normalized timeout fields exist | Mostly done |
| Failure propagation | Failures return `success: False` and `error` where available | Mostly done |
| Cache interactions | Session/cache paths exist, cache hits short-circuit, cache API drift fixed | Mostly done |
| Context interactions | Best-effort state get/set and prompt suggestions exist | Mostly done |
| Tool exposure coherence | FastMCP active surface coherent | Mostly done |
| Route drift removed/documented | Legacy compatibility remains | Not done |

## Recommended Phase 2 Next Actions

1. Add one typed-wrapper-vs-generic equivalence test for normalized output.
2. Document compatibility drift as Phase 1 output before spending time stabilizing legacy `server_api/`.
3. Re-run the full configured suite before declaring Phase 2 closed.

## Bottom Line

Phase 2 is materially improved after the fix pass. The main bad-flow bugs identified in this audit were patched in `run_security_tool(...)`. Remaining work is validation, one wrapper/generic equivalence test, and explicit compatibility-surface documentation from Phase 1.
