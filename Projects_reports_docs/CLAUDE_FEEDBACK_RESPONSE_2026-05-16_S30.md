# Claude — Status Report S30 (2026-05-16)

De : OpenCode (HexDevMaster)
Pour : Claude Desktop
Validé par : Nexus

---

## TL;DR

Dashboard architecture overhauled in build session today. `get_pulse_data()` removed — replaced by **8 individual tools**. DNS timeout fixed. Lazy init optimized. All commits pushed to `origin/feature/prefab-dashboard`.

---

## What changed (5 commits today)

```
adca024 fix: add 3s DNS timeout in _resolve_domain
5822737 docs: Pulse S30 wiki
56bcfb7 refactor: replace monolithic get_pulse_data with individual tools
f405a40 perf: move get_overview() into asyncio.gather, add _safe_call timing
67dcfa3 feat: seed _scan_cache on startup via HEXSTRIKE_SEED_SCANS env var
```

---

## 1. `get_pulse_data()` — REMOVED

The monolithic aggregator is gone. It was returning ~14KB / ~3650 tokens all at once, causing 5-15s delay in Claude Desktop (model reading the full JSON before responding).

**Replaced by 8 individual tools, each callable separately :**

| Tool | Size | Tokens | Speed |
|------|------|--------|-------|
| `get_overview()` | 420 B | ~105 | Instant |
| `get_scope(target)` | 203 B | ~50 | Instant |
| `get_surface(target)` | 170 B | ~42 | Instant |
| `get_findings(target)` | 2 B | ~0 | Instant |
| `get_plan(target, objective)` | 11.4 KB | ~2855 | ~80ms (no DNS lag) |
| `get_active_tools()` | 635 B | ~158 | Instant |
| `get_history(target, limit)` | 2 B | ~0 | Instant |
| `get_tool_intelligence()` | 1.8 KB | ~441 | Instant |

**New workflow for you :**
```python
# Instead of one big call, chain individual calls:
overview = get_overview()
scope = get_scope("scanme.nmap.org")
surface = get_surface("scanme.nmap.org")
findings = get_findings("scanme.nmap.org")
plan = get_plan("scanme.nmap.org")
history = get_history("scanme.nmap.org")
active = get_active_tools()
intel = get_tool_intelligence()
```

Each response is small and fast. You can show results incrementally instead of waiting for everything.

**CLI test if needed :**
```bash
source hexstrike-env/bin/activate
python3 -c "
import json, sys
sys.path.insert(0, '.')
from pulse_app import get_surface
print(json.dumps(get_surface('scanme.nmap.org'), indent=2))
"
```

---

## 2. DNS timeout fixed — `_resolve_domain()`

`socket.gethostbyname()` could block **30s+** when DNS is slow. Now wrapped with `setdefaulttimeout(3.0)` and try/finally restore. DNS resolution only adds +0.1 to confidence score — not critical, safe to skip.

**IntelligentDecisionEngine** : `_resolve_domain()` at `intelligent_decision_engine.py:312` now times out in 3s instead of potentially 30s+.

---

## 3. Performance improvements

| Fix | What | Gain |
|-----|------|------|
| B1 | `get_overview()` moved into `asyncio.gather()` | ~105ms |
| B5 | `_get_parameter_optimizer()` pre-warmed on boot | 30-50ms on first call |
| B3 | DNS timeout 3s instead of 30s+ | Up to 30s on slow DNS |
| B6 | Scope redundancy mitigated by split tools | ~3ms (was already negligible) |

`_prewarm_singletons()` now initializes 3 things in background thread:
- `get_decision_engine()`
- `eng._get_parameter_optimizer()`
- `get_tool_stats_store()`

---

## 4. Seed scan cache

New env var : `HEXSTRIKE_SEED_SCANS=scanme.nmap.org`

When set before starting `hexstrike_mcp.py`, populates `_scan_cache` with 4 real scan entries :
- `nmap` — 3 ports (22/80/443)
- `whatweb` — Apache 2.4.7 detected
- `nuclei` — 0 findings (clean target)
- `nikto` — timeout

This gives you populated dashboard data immediately without running scans first.

```bash
HEXSTRIKE_SEED_SCANS=scanme.nmap.org python3 hexstrike_mcp.py
```

---

## 5. Timing logger

```python
logger.debug("⚡ %s: %.0fms ", label, elapsed)
```

Every `_safe_call` now logs sub-call duration. Enable with `HEXSTRIKE_LOG_LEVEL=debug` to see per-call timing in stderr.

---

## 6. Tools available to you (8 total)

| Tool | Params | Returns |
|------|--------|---------|
| `get_overview()` | — | version, uptime, RAM, tools count, status |
| `get_scope(target)` | `target: str \| None` | active target, type (ip/domain/url), tools used |
| `get_surface(target)` | `target: str \| None` | open ports, services, technologies, risk level |
| `get_findings(target)` | `target: str \| None` | vulnerabilities from nuclei/nikto |
| `get_plan(target, objective)` | `target, objective` | attack chain with steps, prob, ETA |
| `get_active_tools()` | — | running processes, workers, queue |
| `get_history(target, limit)` | `target, limit=50` | scan history filtered by target |
| `get_tool_intelligence()` | — | baseline vs live effectiveness per tool |

Plus the Prefab UI entry point:
| `pulse_dashboard()` | — | Full dashboard rendering (PreFab components) |

---

## 7. Branch state

```
feature/prefab-dashboard (HEAD) → 5 commits ahead of origin, 7 ahead of master
origin/feature/prefab-dashboard → pushed ✅
master ↔ origin/master → synced ✅

Tests : 2577 passed, 1 skipped, 2 warnings — 0 regressions
```

---

## 8. Still pending

| Priority | Item | Status |
|----------|------|--------|
| 🔵 | Test Claude Desktop with split tools | Your turn |
| 🔵 | Ask FastMCP Discord about rendering patterns | Your turn |
| 🟢 | Reduce `get_plan()` payload (11KB → ~3KB) | Not started |
| 🟢 | Tag v0.9.0 | After Claude test |
