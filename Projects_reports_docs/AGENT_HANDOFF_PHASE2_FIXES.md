# Agent Handoff: Phase 2 Bad-Flow Fixes

Date: 2026-05-01  
Scope: fixes applied after `PHASE2_RUNTIME_STABILIZATION_AUDIT.md`.

## Summary

The Phase 2 audit found that `run_security_tool(...)` had inconsistent edge behavior:

- early returns skipped `_op_metrics.record(...)`
- early returns had minimal, non-canonical result shape
- cache-hit returns skipped metrics and did not normalize cached results
- `parameters` annotation said `str`, while typed wrappers passed dicts

These were fixed in `mcp_core/server_setup.py`.

A second Phase 2 bug was found during validation: `server_core/command_executor.py` had been changed to the `AdvancedCache` one-argument API, but the shared singleton cache is `HexStrikeCache`, which requires `(command, params)`. That broke real `nmap` execution through `run_security_tool(...)`. It is fixed with compatibility helpers.

## Code Changes

### `mcp_core/server_setup.py`

Added `_normalize_tool_result(...)`.

It guarantees these keys on returned tool results:

- `success`
- `output`
- `error`
- `returncode`
- `timed_out`
- `partial_results`
- `execution_time`
- `timestamp`

Updated `run_security_tool(...)`:

- `parameters` now accepts `str | Dict[str, Any]`.
- added non-object JSON validation.
- added local `finalize(result)` helper.
- routed invalid JSON through `finalize(...)`.
- routed non-object JSON through `finalize(...)`.
- routed unknown tool through `finalize(...)`.
- routed destructive-denied path through `finalize(...)`.
- routed cache-hit path through `finalize(...)`.
- normalized direct execution result before success/failure handling.
- final execution return now uses `finalize(result)`.

### `server_core/command_executor.py`

Fixed `_normalize_result(...)` minimal-error handling:

- minimal dicts such as `{"success": False, "error": "'target' is required"}` now pass through without losing the error message.

Added cache compatibility helpers:

- `_cache_get(...)`
- `_cache_set(...)`

These support both:

- `AdvancedCache.get(key)` / `AdvancedCache.set(key, value)`
- `HexStrikeCache.get(command, params)` / `HexStrikeCache.set(command, params, result)`

`finalize(...)` now:

- normalizes result shape
- sets telemetry success
- sets telemetry timeout state
- sets telemetry duration
- logs telemetry
- records `_op_metrics`
- returns the normalized result

## Test Changes

### `tests/test_server_setup_standalone.py`

Added bad-flow tests:

- invalid JSON records metrics and returns canonical shape
- non-object JSON records metrics and returns canonical shape
- unknown tool records metrics and returns canonical shape
- destructive denial records metrics
- cache hit records metrics and normalizes cached result

### `tests/test_command_executor_normalize.py`

Added/validated:

- minimal error dicts keep their error message
- `execute_command(...)` supports the legacy `HexStrikeCache` signature

## Remaining Follow-Up

1. Targeted tests were run successfully:

```bash
hexstrike-env/bin/python3 -m pytest \
  tests/test_server_setup_standalone.py \
  tests/test_command_executor_normalize.py \
  tests/test_operational_metrics.py \
  tests/test_fastmcp3_ctx_methods.py \
  tests/test_skills_modernization.py \
  tests/test_fastmcp_context_regressions.py \
  -q
```

Result:

```text
96 passed in 285.19s
```

Full configured suite was also run:

```bash
hexstrike-env/bin/python3 -m pytest tests/ -q
```

Result:

```text
1323 passed, 6 xfailed, 2 warnings in 79.56s
```

2. Add one equivalence test:

- generic `run_security_tool(...)`
- typed wrapper for the same route
- same mocked direct result
- same canonical result shape

3. Re-run the full configured suite before declaring Phase 2 closed.

## Note

Earlier WSL calls intermittently failed with `Wsl/Service/0x8007274c`. If tests do not run, treat that as an environment failure until reproduced inside the WSL shell directly.
