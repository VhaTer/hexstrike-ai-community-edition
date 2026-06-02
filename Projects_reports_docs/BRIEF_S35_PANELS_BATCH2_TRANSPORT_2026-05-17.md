# S35 — Tool descriptions enrichies + Panels Batch 2 (Sessions, Confirmations, Network I/O)

**Date:** 2026-05-17
**Branch:** master (v0.9.0)

---

## 1. Cold start fix — tool descriptions enrichies 🟢

**Problème :** `get_overview`, `get_surface`, `get_findings`, `get_plan` avaient des descriptions trop courtes (1 ligne) — le LLM ne les trouvait pas en session fraîche.

**Fix :** descriptions enrichies avec :
- Usage workflow (quoi utiliser avant/après)
- Structure du retour JSON
- Exemples d'appel
- Hints de découverte (search_tools friendly)

Descriptions passées de ~5 mots à ~40-60 mots chacune. Couche 1 du plug-and-play livrée.

## 2. Sessions panel 🟢

Nouveau `@app.tool() get_sessions()`:
- `SessionStore.list_active()` — sessions actives
- `SessionStore.list_completed()` — sessions archivées (max 20 récentes)
- DataTable : Session / Target / Findings / Age
- Import via `get_session_store()` depuis `server_core.singletons`

## 3. Confirmations panel 🟢

Nouveau `@app.tool() get_confirmations()`:
- `_op_metrics.confirmation_summary()` — accepted/denied/skipped
- Card Metrics : Accepted / Denied / Skipped

## 4. Network I/O panel 🟢

Nouveau `@app.tool() get_network_io()`:
- `ResourceMonitor.usage_history[-1]` — dernier snapshot
- bytes_sent / bytes_recv formatés (B/KB/MB/GB)
- Card Metrics : Sent / Received / Total

## Test result

```
2577 passed, 1 skipped, 2 warnings — 0 regressions
50/50 pulse app tests
```

## Dashboard panels (17 total)

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
| 10 | Tool Performance | `get_tool_performance()` | S34 |
| 11 | Cache Status | `get_cache_status()` | S34 |
| 12 | System Trends | `get_system_trends()` | S34 |
| 13 | **Sessions** | `get_sessions()` | **S35** |
| 14 | **Confirmations** | `get_confirmations()` | **S35** |
| 15 | **Network I/O** | `get_network_io()` | **S35** |
| 16 | Intelligence | `get_tool_intelligence()` | S27 |
| 17 | Footer | `_op_metrics` | S28 |

## Files changed

| File | Change |
|---|---|
| `pulse_app.py` | 3 new tools + 3 panels + Rx + state + enriched descriptions for 4 tools (~200 lines) |

## Remaining

- **Transport audit stdio vs HTTP** — vérifier si `hexstrike_mcp.py` supporte `--transport http`
- **Tag v0.10.0** — après stabilisation S35
- **Discord FastMCP** — abandonné
- **RPI vuln lab** — prochain milestone
