# HexStrike Full Audit — Executive Summary

Generated 2026-05-08 for `feature/attack-intelligence` branch.

## Top 3 Stabilizing Actions

1. **Freeze the README** — change "150+ tools" to "106", wire 8 orphaned handlers, document the 34 hidden tools. Stops promising what doesn't exist.

2. **Delete `mcp_tools/`** — 193 dead files, ~11K LOC, Phase 2 archaeology. Keeping it confuses every developer who reads the codebase. Archive to a zip if history matters.

3. **Add CI** — one `.github/workflows/ci.yml` running `pytest tests/ -q`. Without it, every PR merges blind.

## Bottom Lines

- **Unshipped tools:** 31 listed but not wired (8 have code, 23 don't)
- **Undocumented tools:** 34 work but are invisible to users
- **Dead code:** 209 files / ~16K LOC (193 in `mcp_tools/`, 14 test files, 2 core modules)
- **Branch drift:** 146 ahead / 118 behind upstream — 264 commits of divergence
- **Tests:** 1,091 tests, 99.2% pass rate, but suite timed out at 60s (AGENTS.md says ~13s)
- **Critical missing:** SECURITY.md (pentesting tool with no disclosure policy), CI/CD (no automated checks)

## Changes Applied (2026-05-08)

| Change | Impact |
|---|---|
| **README OpenCode config** — `"type": "http"` → `"type": "remote"` | OpenCode can now connect to HexStrike MCP |
| **BM25SearchTransform optimized** — max_results 5→15, markdown serializer (-40% tokens), 3 pinned tools | Better tool discovery, lower token cost |
| **Full audit written** to `Projects_reports_docs/full_audit.md` | Agent has complete codebase health reference |
| **AGENTS.md created** (gitignored) | Repo-specific instructions for future coding sessions |
