# Code Changes — 2026-05-08

All changes made during the audit and stabilization session, listed for the agent.

---

## 1. README OpenCode Config Fix

**File:** `README.md`  
**Line:** 82  
**Type:** Bug fix

Changed OpenCode MCP server config from invalid `"type": "http"` to correct `"type": "remote"`:

```diff
 {
   "$schema": "https://opencode.ai/config.json",
   "mcp": {
     "hexstrike-pulse": {
-      "type": "http",
+      "type": "remote",
       "url": "http://127.0.0.1:8888/mcp",
       "enabled": true
     }
   }
 }
```

**Why:** OpenCode's config schema only supports `"type": "local"` and `"type": "remote"`. The old value was silently ignored.

---

## 2. BM25SearchTransform Optimization

**File:** `mcp_core/server_setup.py`  
**Lines:** 29, 30-32, 496-502  
**Type:** Enhancement

Three coordinated edits:

### 2a. Import markdown serializer (line 29)

```diff
-    from fastmcp.server.transforms.search import BM25SearchTransform
+    from fastmcp.server.transforms.search import BM25SearchTransform, serialize_tools_for_output_markdown
```

### 2b. Add fallback variable (lines 30-32)

```diff
 except ImportError:
     BM25SearchTransform = None
+    serialize_tools_for_output_markdown = None
```

### 2c. Upgrade transform params (lines 496-502)

```diff
-    transforms = [BM25SearchTransform()] if BM25SearchTransform else []
+    transforms = [BM25SearchTransform(
+        max_results=15,
+        search_result_serializer=serialize_tools_for_output_markdown,
+        always_visible=["nmap", "whatweb", "sqlmap"],
+    )] if BM25SearchTransform else []
```

**Before:**
- `tools/list` returns 2 tools (search_tools, call_tool)
- `search_tools` returns JSON — verbose, ~200 chars/tool
- Max 5 results

**After:**
- `tools/list` returns 5 tools (nmap, whatweb, sqlmap, search_tools, call_tool)
- `search_tools` returns compact markdown — ~115 chars/tool, ~40% fewer tokens
- Max 15 results

---

## 3. Full Audit Report

**File:** `Projects_reports_docs/full_audit.md`  
**Lines:** 573 → 633  
**Type:** New file (with appended changes section)

10-section audit covering:
1. README Inflation — 31 unshipped tools, 34 undocumented
2. Dead Code — 209 files / ~16K LOC
3. Missing Community Infrastructure — 7 gaps
4. Branch Health — 146 ahead / 118 behind upstream
5. Dependency Health — 293 packages, pinned bcrypt/chardet
6. Test Suite Reliability — timeout regression
7. Code Quality — DIRECT_TOOLS as local var, silent try/except
8. Architecture Concerns — single point of failure, no persistence
9. Stabilization Roadmap — 5 phases
10. Quick Wins vs Deep Work
11. Changes Applied — this session's work

---

## 4. Executive Summary

**File:** `Projects_reports_docs/executive_summary.md`  
**Lines:** 20 → 28  
**Type:** New file (with appended changes section)

Short-form audit with top 3 stabilizing actions, bottom-line numbers, and completed changes log.

---

## 5. AGENTS.md (gitignored)

**File:** `AGENTS.md`  
**Type:** New file, 80 lines

Repo-specific OpenCode instructions. Covers: setup, entrypoints, architecture, testing commands, web UI, constraints, contributing conventions.

---

## 6. Health Bug Fixes (previous session — context)

**File:** `hexstrike_server.py:86`  
**Change:** Hardcoded `"healthy"` → dynamic `"healthy" if all_essential_available else "degraded"`

**File:** `mcp_core/server_setup.py:1052-1085`  
**Change:** Enriched `health://server` resource with `op_metrics` and `type_count`; registered `server_health` MCP tool.

**Test files:**
- `tests/test_hexstrike_server_routes.py` — 3 new tests
- `tests/test_server_setup_standalone.py` — 3 new tests

All 36 tests pass from that session.

---

## Summary of All Changed Files

| File | Change | Type |
|---|---|---|
| `README.md:82` | `"type": "http"` → `"type": "remote"` | Fix |
| `mcp_core/server_setup.py:29` | Added `serialize_tools_for_output_markdown` import | Enhancement |
| `mcp_core/server_setup.py:30-32` | Added fallback var | Enhancement |
| `mcp_core/server_setup.py:496-502` | BM25 params: max_results, markdown, always_visible | Enhancement |
| `AGENTS.md` | Created (gitignored) | New |
| `Projects_reports_docs/full_audit.md` | Created + section 11 appended | New |
| `Projects_reports_docs/executive_summary.md` | Created + changes section appended | New |
| `Projects_reports_docs/code_changes.md` | Created (this file) | New |

### Previous session (health bugs):
| `hexstrike_server.py:86` | Dynamic health status | Fix |
| `mcp_core/server_setup.py:1052-1085` | Health resource + tool | Fix |
| `tests/test_hexstrike_server_routes.py` | 3 test cases | Test |
| `tests/test_server_setup_standalone.py` | 3 test cases | Test |
