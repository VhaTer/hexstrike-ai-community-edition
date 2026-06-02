# S34 — Cache seed fix + Panels Batch 1

**Date:** 2026-05-17
**Branch:** master (v0.9.0)
**Commits:** on master, ahead of origin

---

## 1. Cache seed cache-hit serving — FIXED 🟢

**Problem:** Seed entries stored as `seed:{tool}:{target}:{timestamp}` could never be found by `run_security_tool()` which builds lookups as `{session}:{tool}:{target}[:{hash}]` via `_cache_key_for()`.

**Fix (2 changes):**
- `mcp_core/mcp_entry.py:46` — key format: `f"seed:{tool}:{seed_target}"` (removed `:{int(now)}` timestamp suffix)
- `mcp_core/server_setup.py:978-983` — fallback lookup after direct cache miss: `_scan_cache.get(f"seed:{tool_name}:{target}")`

**Validation:** Seed entries now serve as cache hits. In a fresh session:
```
⚡ nmap seed cache hit for scanme.nmap.org
```

## 2. Tool Performance panel 🟢

New `@app.tool() get_tool_performance()`:
- `_op_metrics.success_rate_by_tool()` — worst-first sorted
- `_op_metrics.timeout_count_by_tool()`
- Combined DataTable: Tool / Rate / Runs / Timeouts
- Panel between Errors & Failures and Cache Status

## 3. Cache Status panel 🟢

New `_op_metrics.cache_hits_by_tool()` method in `operational_metrics.py`:
- Returns per-tool cache hit counts (tools with hits > 0, sorted desc)

New `@app.tool() get_cache_status()`:
- `cache_summary()` hits/misses/ratio
- `HexStrikeCache.get_stats()` size/max_size/utilization
- `cache_hits_by_tool()` per-tool DataTable
- Metric card row: Hits / Misses / Hit ratio / Size / Max / Util

## 4. System Trends panel 🟢

New `@app.tool() get_system_trends()`:
- `ResourceMonitor.get_usage_trends()` — CPU avg 10, mem avg 10
- `usage_history[-30:]` — CPU + memory history for sparklines
- Metric card: CPU avg / MEM avg / Period / Measurements
- Two sparklines (CPU info-blue, memory warning-orange)

## Test result

```
2577 passed, 1 skipped, 2 warnings — 0 regressions
50/50 pulse app tests
119 seed/cache tests passed
```

## Files changed

| File | Change |
|---|---|
| `mcp_core/mcp_entry.py` | seed key: drop `:{timestamp}` suffix |
| `mcp_core/server_setup.py` | fallback `_scan_cache.get(f"seed:{tool}:{target}")` |
| `server_core/operational_metrics.py` | new `cache_hits_by_tool()` method |
| `pulse_app.py` | 3 new tools + 3 panels + Rx + state (~250 lines) |

## Dashboard panels (14 total)

| # | Panel | Source | Since |
|---|---|---|---|
| 1 | Header (Icon+Sparkline) | `get_overview()` | S32 |
| 2 | Scope | `get_scope()` | S28 |
| 3 | Surface | `get_surface()` | S29 |
| 4 | Findings | `get_findings()` | S29 |
| 5 | Plan IDE | `get_plan()` | S29 |
| 6 | Active Tools | `get_active_tools()` | S29 |
| 7 | History | `get_history()` | S29 |
| 8 | Rate Limit | `get_rate_limit_status()` | S31 |
| 9 | Errors & Failures | `get_errors_and_failures()` | S33 |
| 10 | **Tool Performance** | `get_tool_performance()` | **S34** |
| 11 | **Cache Status** | `get_cache_status()` | **S34** |
| 12 | **System Trends** | `get_system_trends()` | **S34** |
| 13 | Intelligence | `get_tool_intelligence()` | S27 |
| 14 | Footer | `_op_metrics` | S28 |

## Remaining for S35

- Sessions panel (SessionStore data)
- Confirmations panel (accepted/denied/skipped)
- Network I/O panel (bytes sent/recv)
- Named MCP wrappers `get_overview`/`get_surface`
- Transport audit stdio vs HTTP
