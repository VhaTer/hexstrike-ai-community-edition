# HexStrike Stabilization Session Report

Date: 2026-05-01  
Session scope: Phase 1 → Phase 3 of the stabilization plan  
Branch: `feature/attack-intelligence`  
HEAD at session start: `4cd532b`  
HEAD at session end: `b1a8436`  
Author: Claude (Anthropic) + HexReaper

---

## Context

This session applied the stabilization plan (`hexstrike-ai_pulse_stabilisation_plan.Md`) starting from
the push `4cd532b..b1a8436`. Work covered Phase 1 (freeze / triage), Phase 2 (runtime stabilization),
and Phase 3 (workflow audit). Phase 4 (observability) was started and is in progress.

---

## Bugs Fixed This Session

### P0 — `plan_attack`: `chain` variable never defined

**File:** `mcp_core/server_setup.py`

`chain` was referenced in `return chain.to_dict()` but was only defined inside an `if` branch that was
never reached. After the `if profile is None / else` block, `create_attack_chain()` was never called.
Result: `NameError` on 100% of `plan_attack` calls.

**Fix:** Moved `chain = await loop.run_in_executor(None, lambda: _ide.create_attack_chain(profile, objective))`
to after both branches, so it always executes regardless of whether the profile came from session state
or fresh analysis. Added `ctx.report_progress(100)` and success log before return.

---

### P0 — `TargetProfile.from_dict()` missing

**File:** `shared/target_profile.py`

`plan_attack` calls `TargetProfile.from_dict(saved_profile)` when restoring from session state.
The classmethod did not exist — only `to_dict()` was defined. Result: `AttributeError` on every
session-restore path in `plan_attack`.

**Fix:** Added `from_dict()` classmethod with full deserialization: target type enum recovery with
`ValueError` fallback to `UNKNOWN`, `TechnologyStack` enum recovery per-item with skip on unknown,
and all scalar fields with safe defaults.

---

### P0 — `execute_command` result shape mismatch (`output`/`error` vs `stdout`/`stderr`)

**File:** `server_core/command_executor.py`

`EnhancedCommandExecutor.execute()` returns `{stdout, stderr, return_code, success, timed_out, ...}`.
All consumers — `run_security_tool`, `RateLimitDetector`, `_scan_cache`, `_detect_from_cache` — read
`result.get("output")` and `result.get("error")`. These always returned `None`/`""` silently, meaning:

- Rate limit detection was blind (always saw empty string)
- Error logging always said `unknown`
- Scan cache stored results but consumers couldn't read them

**Fix:** Added `_normalize_result()` in `command_executor.py`. Called on every `execute()` return.
Produces canonical shape: `{success, output, error, returncode, timed_out, partial_results,
execution_time, timestamp}` plus legacy aliases `{stdout, stderr, return_code}` for backward compat.

Normalization rules:
- Already-normalized dicts (have `output`, no `stdout`) pass through unchanged — `_require()` early
  returns are safe
- `output` = `stdout` if non-empty, else `stderr` (some tools write results to stderr)
- `error` = `stderr.strip()` on failure, or timeout message when `timed_out=True`, else `""`
- Timeout error message includes `execution_time` and first 200 chars of stderr

---

### P1 — `active_cache.get()` called with wrong arity

**File:** `server_core/command_executor.py`

`active_cache.get(command, {})` — `AdvancedCache.get()` takes one argument. The second positional
argument `{}` would cause a `TypeError` on any cache hit attempt.

**Fix:** `active_cache.get(command)` — one argument only.

---

### P1 — `active_cache.set()` storing empty dict instead of result

**File:** `server_core/command_executor.py`

`active_cache.set(command, {}, result)` — three positional args. `AdvancedCache.set(key, value, ttl)`.
This stored `{}` as the cached value and passed the entire `result` dict as `ttl`, which would raise
a `TypeError` in `_ScanCache.set()` at `if execution_time > 60` because `execution_time` would be
the result dict.

**Fix:** `active_cache.set(command, result)` — stores the actual result, uses default TTL.

---

## Tests Added

### `tests/test_command_executor_normalize.py` — 15 tests

New file. Covers `_normalize_result()` and `execute_command()` integration:

| Test class | Cases |
|---|---|
| `TestNormalizeResult` | passthrough (already-normalized), passthrough (_require early-return), success stdout→output, success empty-stdout fallback to stderr, failure stderr→error, failure no-stderr empty error, timeout success=False, timeout with stderr appended, timeout stderr truncated to 200 chars, legacy keys preserved, all canonical keys present, missing optional fields default gracefully |
| `TestExecuteCommandNormalization` | execute_command returns canonical shape (mocked executor), execute_command timeout normalized (mocked executor), execute_command cache stores result not empty dict (mocked cache) |

---

## Phase 3 Workflow Audit Results

All 26 tools referenced across the 5 workflow prompts were verified against `DIRECT_TOOLS` in
`mcp_core/server_setup.py`. Zero tools referenced in prompts are missing from the runtime surface.

| Prompt | Tools | Status |
|---|---|---|
| `bug_bounty_recon` | wafw00f, httpx, subfinder, amass, rustscan, nmap, katana, gobuster, nuclei, nikto | ✅ all present |
| `wifi_attack_chain` | airmon_ng, airodump_ng, aireplay_ng (gated), aircrack_ng | ✅ all present |
| `ctf_web_challenge` | wafw00f, httpx, nikto, gobuster, ffuf, katana, nuclei, sqlmap, dalfox | ✅ all present |
| `smb_lateral_movement` | nbtscan, nmap, enum4linux, smbmap, netexec, metasploit (gated) | ✅ all present |
| `cloud_security_audit` | prowler, trivy, kube_hunter | ✅ all present |

**P2 noted (deferred):** `nuclei` is routed through `misc_exec` in `DIRECT_TOOLS` but mapped to
`"web-vuln"` in `_TOOL_SKILL_MAP`. Skill hint works correctly, category placement is inconsistent.
No runtime impact.

**P2 noted (deferred):** `_tool_stats.record()` is not called from `run_security_tool` on the
standalone path. `blended_effectiveness()` scores remain static-baseline-only. No runtime impact,
but IDE tool selection quality is not improving from live feedback.

---

## Deferred Items (P2)

| Item | File | Notes |
|---|---|---|
| `nuclei` in `misc_exec` vs `web-vuln` skill | `server_setup.py` | Cosmetic only |
| `_tool_stats.record()` not called in `run_security_tool` | `server_setup.py` | IDE effectiveness data stays static |
| `tool_stats_store.record()` docstring says success requires `stdout` non-empty | `tool_stats_store.py` | Doc inconsistency; `record()` takes a bool, not a result dict |

---

## Phase 4 Status (In Progress)

The existing `[telemetry]` JSON log in `run_security_tool` already captures:

```
tool, success, duration, timeout, cache_hit, session_state,
confirmation (None/accepted/denied/skipped), opt_profile,
skill_injected, prompt_suggested
```

**Missing from current telemetry vs Phase 4 spec:**

| Missing field | Spec requirement |
|---|---|
| `target` (normalized) | "normalized target identity where safe" |
| `timed_out` | "timeout state" — present in result but not copied to telemetry |
| `partial_results` | implicit from timeout but not explicit |
| `cache_hit` is always `False` | flag is declared but never set to `True` |
| `session_state` is always `False` | flag is declared but never set to `True` |
| `typed_tool_source` | "typed tool vs generic tool invocation source" |

**`cache_hit` and `session_state` are declared but never populated** — this is the next fix target
in Phase 4.

---

## Next Steps

1. **Phase 4 — fix `cache_hit` and `session_state` flags in telemetry** (next immediate fix)
2. **Phase 4 — add `target` (normalized) and `timed_out` to telemetry dict**
3. **Phase 4 — add operational log views** (`error_count_by_tool`, `timeout_count_by_tool`,
   `success_rate_by_tool`) — likely a new `server_core/telemetry_store.py` backed by the existing
   `TelemetryCollector` or a new structured JSON store
4. **Phase 5 — regression validation pass** on `run_security_tool` integration, cache paths,
   destructive confirmation, session restore
5. **Commit this session's work** before continuing

---

## Commit Message (suggested)

```
fix: P0/P1 runtime stabilization — normalize result shape, fix plan_attack chain, fix cache API

- Add _normalize_result() to command_executor: canonical {output, error, returncode}
  shape from EnhancedCommandExecutor {stdout, stderr, return_code}
- Fix execute_command cache: .get() one arg, .set(command, result) not (command, {}, result)
- Add TargetProfile.from_dict() classmethod — required by plan_attack session restore
- Fix plan_attack: create_attack_chain() called after if/else, not inside dead branch
- Add 15 regression tests: test_command_executor_normalize.py

Bugs fixed: 2×P0, 2×P1
Tests: 1277 → 1292 (all passing, 6 xfailed)
Phase: stabilization/phase2-phase3 complete
```
