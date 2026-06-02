# Session 70-71 — Bridge notification fix + Multi-client HTTP architecture + v0.11.0

**Date** : 2026-06-01
**Tag** : `v0.11.0` (commit `98cdeca`, branch `master`)

## Résumé

Migration vers une architecture multi-client : 1 serveur HTTP unique (`:8888`), tous les clients partagent la même instance. Claude Desktop connecté via un bridge stdio→HTTP léger (stdlib only, 0.3s init). La cause racine des échecs Claude Desktop identifiée et corrigée (réponse aux notifications MCP sans `id`). CTF dashboard réparé, `pulse_dashboards_guide` rendu visible. Tag v0.11.0 pushé, GitHub Wiki enrichi.

## Bug corrigé — Bridge répondait aux notifications

**Fichier** : `pulse-bridge.py`

**Symptôme** : Claude Desktop rejetait la connexion avec `invalid_union` sur le champ `id` — ni string ni number.

**Cause** : Le bridge envoyait `{"id": null, "result": {}}` pour `notifications/initialized` (JSON-RPC notification, pas de `id`). Claude Desktop valide chaque message JSON-RPC et rejette `id: null`.

**Fix** : `is_notification = req.get("id") is None` → skip toute réponse pour les notifications.

## SSE parser rewrite

**Fichier** : `pulse-bridge.py`

- Split sur `\n\n` (boundary d'événement SSE) au lieu de per-line → gère correctement les valeurs `data:` multi-lignes
- `http.client.HTTPConnection` persistent (keep-alive) avec auto-retry sur connexion stale → remplace `urllib` (fresh connection chaque requête)
- `ensure_ascii=False` pour l'output JSON (unicode-safe)

## Architecture multi-client

```
┌──────────────────────────────────────────────────┐
│                  WSL2 (Kali Linux)                │
│                                                    │
│   ┌─────────────────┐     ┌──────────────────┐   │
│   │  HTTP Server     │     │  pulse-bridge.py  │   │
│   │  :8888           │◄────│  stdio→HTTP       │   │
│   │  hexstrike_mcp   │     └────────┬─────────┘   │
│   │                  │              │              │
│   └────────┬────────┘              │              │
│            │                       │ wsl.exe       │
│            │ remote URL            │               │
│            │                       ▼               │
│            │              ┌──────────────────┐   │
│            │              │  Claude Desktop   │   │
│            │              │  (Windows)        │   │
│            │              └──────────────────┘   │
│            │                                      │
│            ▼                                      │
│   ┌──────────────────┐                            │
│   │  opencode/        │                            │
│   │  Continue/Cline   │                            │
│   └──────────────────┘                            │
└──────────────────────────────────────────────────┘
```

- **1 serveur HTTP** sur `127.0.0.1:8888` — tous les clients partagent la même instance
- **Claude Desktop** via `--bridge` (`wsl.exe -e python3 -u pulse-bridge.py`) — bash retiré, startup messages éliminés
- **Opencode** : `type: "remote"`, `url: http://localhost:8888/mcp`
- **Continue/Cline** : même URL en remote/streamable-http
- Zéro conflit lock file, zéro timeout d'import fastmcp (bridge stdlib only)

## Changements fichiers

### Nouveaux fichiers

- `pulse-bridge.py` — stdio→HTTP proxy, stdlib only (json + http.client). Init 0.3s. Gère sessions MCP, capture `mcp-session-id` header, parsing SSE complet.

### Fichiers modifiés

- `hexstrike-pulse` — Launcher 5 modes (start/stop/status/foreground/bridge/stdio). HTTP transport par défaut. WSL detection + IP display.
- `pulse_app.py` — 2 fixes :
  - CTF dashboard : `_cat_tool_list` flattening — `category_tools` était `dict[str, dict[str, list[str]]]`, pas `dict[str, list[str]]`. Le slice `cats.get(cat, [])[:8]` throwait `TypeError: unhashable type: 'slice'`.
  - `pulse_dashboards_guide` : `@app.tool()` → `@app.tool(model=True)` — l'agent MCP peut maintenant découvrir et appeler le guide des dashboards.
- `README.md` — Documentation multi-client workflow complète
- `AGENTS.md` — Session 70-71 notée
- `opencode.json` — `type: "remote"`, `url: http://localhost:8888/mcp`

## CTF dashboard validé

Appelé via `call_tool("ctf_dashboard", {})` retourne `structuredContent` Prefab UI complet :

```
Icon(swords) · Heading("CTF Challenge Dashboard") · Badge("7 categories")
→ Categories & Tools: crypto(14), web(12), forensics(10), osint(8), ...
→ BarChart: Tool Coverage by category
```

## search_tools validé

Les résultats sont dans `content[0].text` (wrap_result), pas dans `result.result` :

```
search_tools("ctf")       → 7 tools dont ctf_dashboard
search_tools("dashboard") → 3 tools (get_overview, ctf_dashboard, pulse_dashboards_guide)
search_tools("pentest")   → 4 tools dont pentest_report
search_tools("recon")     → 10 tools (nmap, autorecon, bbot, ...)
search_tools("scan")      → 10 tools (arp_scan, dalfox, dirb, ...)
```

Tous les outils hors `always_visible` sont découvrables via `search_tools()` — plus besoin de pré-charger 160 tools dans `list_tools()`.

## Validation tools vs cible réelle

| Métrique | Valeur |
|----------|--------|
| Tools dans le registry | 162 |
| Disponibles (installés) | 90 |
| Manquants (non installés) | 41 |
| Pulse App (scan, dashboard...) | 31 |
| Testés sur cible web réelle | 7 |

### Tools validés sur web app

| Tool | Résultat |
|------|----------|
| `nmap` | 2 ports ouverts (SSH 22, HTTP 80), TCP connect scan |
| `whatweb` | Apache HTTPD, PHP détectés |
| `nuclei` | Findings critical/medium/info parsés (ANSI strip fix) |
| `nikto` | 12 findings info, timeout 480s (fix -nocheck) |
| `gobuster` | Paths standards découverts (/config/, /about.php) |
| `sqlmap` | Banner MySQL récupéré via injection |
| `dalfox` | XSS réfléchi confirmé |

### Pipeline scan

- `scan(quick)` → nmap + whatweb → surface (ports + technos) — ~30s
- `scan(medium)` → + nuclei + nikto → findings enrichis (Layer 2 score) — ~2-3min
- `scan(full)` → + gobuster → plan + surface + findings — ~5-10min
- `next_suggested_tool` fonctionnel (hydra pour password leak, sqlmap pour SQLi)

## GitHub Wiki

Nouvelle page **[Tool Validation Report](https://github.com/VhaTer/hexstrike-ai-community-edition/wiki/Tool-Validation-Report)** :

- Rapport de validation 7/162 tools
- Tableau détaillé par outil (ce qui est testé, résultat)
- Pipeline scan quick → full
- Matrice de lab recommandée par catégorie (web, AD, WiFi, cloud, forensics)
- Zéro référence interne — contenu purement pentester/utilisateur final

## Tests

```
Bridge                     : 4/4 ✅ (init, notification skipped, tools/list, ping)
pulse_app                  : 96/96 ✅
Core suite (non-slow)      : 2915+ ✅
```

## État du projet

- Architecture multi-client : **✅ stable et documentée**
- Bridge Claude Desktop : **✅ fonctionnel** (0.3s init, notifications filtrées)
- Découvrabilité tools : **✅ search_tools fonctionnel** (wrap_result compris)
- CTF dashboard : **✅ réparé** (navigation + BarChart)
- Documentation utilisateur : **✅ Wiki Tool Validation Report**
- Tag `v0.11.0` pushé, `feature/prefab-dashboard` rebasée sur master
- Prochaine étape : tests avec tokens Claude Desktop, lab multi-cible pour couvrir les 132 tools non testés
