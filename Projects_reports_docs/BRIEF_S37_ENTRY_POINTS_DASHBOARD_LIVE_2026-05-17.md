# Brief S37 — Phase 1 entry points unifiés + Phase 4 dashboard live

**Date:** 2026-05-17  
**Branch:** master  
**Pre-session:** 2577 passed, 1 skipped, 2 warnings — 0 regressions  
**Pre-session panels:** 18

---

## Contexte

S36.5 a livré les async tools (run_async_tool / get_scan_status) pour contourner le timeout 300s de Claude Desktop en stdio. Le dashboard a 18 panels mais l'interface de Claude reste fragmentée : 18 tools séparés à appeler un par un.

## Objectif S37

Deux livrables pour réduire la friction Claude → HexStrike :

### Phase 1 — Entry points unifiés

Créer des points d'entrée haut niveau qui remplacent les appels multiples :

| Entry point | Remplace | Réduction |
|---|---|---|
| `scan(target, intensity, objective)` | get_scope + get_surface + get_findings + get_plan + exécution des tools | ~10 appels → 1 appel |
| `get_live_dashboard(target)` | 15+ tools get_* individuels | 15 appels → 1 appel |

### Phase 4 — Dashboard live

`get_live_dashboard(target=None)` est le livrable central :
- Agrège les 15+ sources de données du dashboard en un seul JSON
- Réutilise `_collect_dashboard_state()` extraite de `pulse_dashboard()`
- Permet à Claude de tout récupérer en 1 appel MCP (~100ms grâce à l'async déjà en place)

## Contre 0.9

- `get_pulse_data(target)` existait dans S30 mais a été retiré. `get_live_dashboard()` est plus complet (18 sources vs 10).
- Les tool descriptions enrichies de S36 (Couche 1) sont conservées.

## Implémentation

### `scan(target, intensity, objective)`

Intensity levels :
- `quick` (défaut) : nmap top 1000 + whatweb → surface analysis
- `medium` : + nuclei + nikto → surface + findings
- `full` : + gobuster (si web) + plan → tout

Workflow :
1. Résout la target
2. Lance les tools appropriés (via run_security_tool)
3. Agrège scope + surface + findings + plan
4. Retourne `{target, intensity, surface, findings, plan, summary}`

### `_collect_dashboard_state(target=None)`

Refactor de la phase data collection de `pulse_dashboard()` (actuellement lignes 1109-1187). Devient une fonction réutilisable :
- Appelée par `pulse_dashboard()` pour alimenter le PrefabApp
- Appelée par `get_live_dashboard()` pour retourner le JSON

## Risques

- `_collect_dashboard_state()` est synchrone → le dashboard UI appelle tout séquentiellement. Si get_live_dashboard est utilisé fréquemment, il faudra le rendre async. Pas urgent.
- scan() exécute des vrais tools (nmap, etc.) — le cache doit les accélérer.

## Post-S37

Le milestone RPI vuln lab peut commencer :
- Déploiement du serveur HexStrike sur Raspberry Pi
- Tests d'intégration avec le lab VulnHub/RPI
- Validation des timeout réseau

---

**State : Proposal**
