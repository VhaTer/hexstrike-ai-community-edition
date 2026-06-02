# Bilan S36.5 + S37 — Async tools + Live dashboard + Entry points unifiés

**Date :** 2026-05-17
**Tag :** v0.10.0 (pre-RPI)
**Tests :** 2577 passed, 1 skipped, 2 warnings — 0 regressions

---

## Résumé

Deux sessions consécutives pour éliminer le goulot d'étranglement principal (timeout 300s Claude Desktop) et réduire la friction d'appel de 15+ tools à 1-2 appels maximum.

## S36.5 — Async tools (Phase 3)

### Problème

Claude Desktop = stdio only. Pas de HTTP/SSE. Timeout 300s incompressible sur les appels MCP longs (sqlmap, nikto, nmap full port scan).

### Solution

- `run_async_tool(tool, target, params)` → retourne `scan_id` en ~2s, exécute en background thread
- `get_scan_status(scan_id)` → polling instantané (0s), pas de timeout
- Cache `_DIRECT_TOOLS_CACHE` + `get_direct_tools()` dans `mcp_core/server_setup.py`

### Livré

- 2 nouveaux tools MCP
- 1 panel Async Scans (running + completed DataTables)
- 18 panels dashboard

## S37 — Entry points unifiés + Dashboard live (Phase 1 + 4)

### Problème

18 tools séparés à appeler un par un. Claude doit enchaîner 15+ appels pour avoir une vue d'ensemble.

### Solution

- `_collect_dashboard_state()` → extraction de la pipeline de données, partagée entre UI et tool
- `get_live_dashboard(target)` → Phase 4 : 18 panels en 1 appel JSON (~100ms)
- `scan(target, intensity, objective)` → Phase 1 : 3 niveaux (quick/medium/full) avec respect du cache

### Livré

- 2 nouveaux tools MCP (20 total)
- `TOOLS_BY_INTENSITY` mapping module-level
- Refactor complet de `pulse_dashboard()` pour réutiliser `_collect_dashboard_state()`

## Métriques

| Métrique | S36 | S37 |
|---|---|---|
| Tools MCP | 18 | 20 |
| Panels dashboard | 18 | 18 |
| Appels Claude → full state | 15+ | 1 |
| Appels Claude → full scan | 10+ | 1 |
| Tests | 2577 | 2577 |

## Prochain milestone : RPI vuln lab

La stack est prête. Le test de référence :

1. Session fraîche Claude Desktop
2. `scan("<lab_ip>", "quick")` → découverte
3. `get_live_dashboard()` → monitoring
4. `run_async_tool("sqlmap", ...)` → deep attack sans timeout

**RPI :** déploiement serveur → intégration VulnHub → validation timeout réseau → documentation.
