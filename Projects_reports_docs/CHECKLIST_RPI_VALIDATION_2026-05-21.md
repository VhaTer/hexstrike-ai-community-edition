# RPI Validation Checklist — Pipeline final

## Prérequis
- [ ] RPI allumé, accessible sur le LAN (IP statique)
- [ ] DVWA installé et fonctionnel (`http://<rpi-ip>/dvwa`)
- [ ] Juice Shop ou WebGoat (optionnel)
- [ ] Client HexStrike prêt (`source hexstrike-env/bin/activate`)
- [ ] `hexstrike-pulse` ou `hexstrike_mcp.py` disponible

---

## Session 1 — Pipeline de base

| Test | Action | Critère |
|---|---|---|
| `scan("quick")` | `scan(target="IP", intensity="quick")` | nmap + whatweb, ports 80/443 |
| `scan("medium")` | intensity="medium" | + nuclei + nikto, ≥1 finding |
| `scan("full")` | intensity="full" | + gobuster + plan d'attaque |
| Cache hit | Re-scanner même cible | Tools "cached", instantané |
| `scan_background("full")` | Appel async | task_id immédiat |

## Session 2 — Findings et parsing

| Test | Critère |
|---|---|
| `get_surface(ip)` | Ports ok, techs détectées | risk level cohérent |
| `get_findings(ip)` | ≥1 vuln nuclei + ≥1 info nikto |
| `get_plan(ip, "exploit")` | Steps générés, proba > 0 |
| TargetStore `target://{ip}/findings` | Findings persistés |
| `next_suggested_tool` | Tool cohérent (sqlmap si SQLi) |

## Session 3 — Dashboard

| Test | Critère |
|---|---|
| `get_live_dashboard(ip)` | 18 panels, données réelles, <200ms |
| Cache Intelligence panel | TTL scores hits > 0 |
| Tool Performance / Intelligence | Données cohérentes |
| Prefab UI `fastmcp dev apps pulse_app.py` | Rendu navigateur |

## Session 4 — Edge cases

| Test | Critère |
|---|---|
| RPI éteint → `scan()` | Timeout propre, pas de crash |
| RPI éteint → `scan_background()` | Task "failed" |
| Cache vidé entre runs | Pas de stale data |
| 3 scans concurrents | Pipeline tient la charge |
| HTTP transport | `hexstrike_mcp.py --transport http` → tools/list OK |

---

## Ce qu'on apprendra

- [ ] nuclei détecte-t-il les vulns DVWA ?
- [ ] Parsing nikto (`+ /admin: Admin login page`) remonte-t-il correctement ?
- [ ] sqlmap est-il proposé comme next tool après un finding SQLi ?
- [ ] Temps réel d'un `full` scan (5 tools) ?
- [ ] Plan d'attaque pertinent sur cible réelle ?
