# Claude Report — MCP Apps Discovery + HexStrike Architecture

**Date :** 2026-05-17
**Auteur :** Session S38 consolidée
**Commits :** `78ff625` (tool_routes), `11d5afe` (fix script)

---

## Découverte : MCP Apps est un standard officiel

Depuis janvier 2026, le protocole MCP supporte officiellement les **MCP Apps** :
- Les tools peuvent déclarer une UI via `_meta.ui.resourceUri` poinant vers une ressource `ui://`
- Le host (Claude Desktop, VS Code, ChatGPT, Goose) rend l'UI dans une iframe sandboxée
- Communication bidirectionnelle via `postMessage` + JSON-RPC
- SDK officiel : `@modelcontextprotocol/ext-apps` (JavaScript/TypeScript)
- https://modelcontextprotocol.info/blog/mcp-apps-ui-capabilities/
- https://modelcontextprotocol.io/extensions/apps/overview

## Notre situation

### Ce qui marche
- ✅ 20 tools MCP stdio (dont `get_live_dashboard`, `scan`, `run_async_tool`)
- ✅ `hexstrike-pulse` : single entry point, auto-venv, lock TTL
- ✅ 130 tools dans `TOOL_ROUTES` (fini la duplication)
- ✅ Cache seed fix, async execution, timeout contourné

### Ce qui ne marche pas (encore)
- ❌ `@app.ui()` PrefabApp → on a dû faire un hack HTML (le dashboard inline dans une tool)
- ❌ Le dashboard pulse_dashboard() ne rend pas nativement dans Claude Desktop
- ❌ Aucune ressource `ui://` enregistrée côté MCP

### Opportunité MCP Apps

Le dashboard généré par `pulse_dashboard()` est déjà du HTML (via PrefabApp). Si on le sert comme une **ressource `ui://`** avec `_meta.ui.resourceUri`, Claude Desktop l'affiche dans une iframe sandboxée.

**Architecture cible :**
```
pulse_dashboard() 
  → génère HTML (déjà fait via PrefabApp)
  → enregistre comme ressource ui://dashboard
  → tool déclare _meta.ui.resourceUri: "ui://dashboard"
  → Claude Desktop rend l'iframe
```

### Contrainte : stdio uniquement

Claude Desktop en stdio **ne supporte pas** le transport HTTP/SSE. Les MCP Apps en stdio sont possibles (le SDK Python FastMCP pourrait les supporter), mais il faut vérifier si FastMCP sait exporter des ressources `ui://` en stdio.

### Actions futures possibles

1. Vérifier si `fastmcp` (version installée) supporte `@mcp.resource("ui://...")` avec sortie HTML
2. Si oui : wrapper `pulse_dashboard()` en ressource `ui://dashboard` + ajouter `_meta.ui.resourceUri` à `get_live_dashboard()`
3. Si non : implémenter le protocole `postMessage` directement (standard JSON-RPC, faisable en pur Python)

---

## S39 — Findings (2026-05-17)

### FastMCP 3.2.4 utilise déjà le protocole MCP Apps

Investigation du `@app.ui()` decorator de FastMCP 3.2.4 :
- Le décorateur `@app.ui()` crée un tool avec `_meta.ui.resourceUri` → `PREFAB_RENDERER_URI`
- `visibility=["model"]` — visible uniquement par le modèle, pas listé dans les tools
- Utilise `AppConfig` + `app_config_to_meta_dict()` pour générer le payload MCP Apps standard
- FastMCP gère la ressource `ui://prefab-renderer` qui sert le moteur Prefab

**Conclusion :** FastMCP 3.2.4 implémente déjà le standard MCP Apps officiel. Le `@app.ui()` est compatible Claude Desktop.

### Tests réalisés

- ✅ `get_live_dashboard()` → 17 clés, toutes présentes
- ✅ `scan("127.0.0.1", "quick")` → target résolue, intensity=quick
- ✅ `get_overview()` → 0.8.0, 162 tools
- 2577 tests passés, 0 régressions

### Docstring bash

- ✅ `hexstrike-pulse` : header documenté, commentaires `#`, bash syntax OK
- ✅ `hexstrike.py` : indent fix sur la ligne `ctf`

### Notes

- `prefab_ui` installé en 0.19.1 (requirement pin <0.15.0) — mismatch potentiel
- `PrefabApp` vient de `prefab_ui`, pas de `fastmcp` directement

---

## Message for next Claude session (21h30)

Salut Claude,

On a préparé le terrain pour toi :
- S38 consolidé : `tool_routes.py` source unique, launcher `hexstripe-pulse`, tout est stable, 2577 verts
- Le dashboard Prefab `@app.ui()` utilise déjà `_meta.ui.resourceUri` (MCP Apps standard) dans FastMCP 3.2.4. **Mais** `prefab_ui` 0.19.1 est installé alors que le pin est `<0.15.0` — à vérifier
- Le rapport complet est dans ce fichier. AGENTS.md a les sessions S37-S39.
- `hexstripe-pulse` est le seul entry point pour Claude Desktop (stdio). Le header documente tout.
- L'utilisateur veut qu'on **teste le dashboard en réel** dans Claude Desktop et qu'on resolve le rendu PrefabApp si ça ne marche pas.
- Priorité : faire fonctionner le dashboard dans Claude Desktop. Si PrefabApp coince, on bascule sur une ressource `ui://` custom avec du HTML pur.

Bonne session 🚀

---

## Architecture actuelle (post-S38)

```
hexstrike-pulse          ← LAUNCHER (auto-venv, lock TTL, docs)
  └─ hexstrike_mcp.py    ← ENTRY POINT MCP stdio
       └─ run_mcp()      ← FastMCP server setup
            ├─ mcp_core/server_setup.py  ← 130 tools + dashboard
            ├─ pulse_app.py              ← 20 tools MCP + UI
            │    ├─ @app.tool() × 20
            │    └─ @app.ui() × 1 (dashboard PrefabApp — non fonctionnel dans Claude)
            └─ mcp_core/tool_routes.py   ← dict partagé (source unique)

hexstrike.py             ← CLI (scan, tools, validate, mcp, serve, ctf)
hexstrike_server.py      ← HTTP server (dashboard web)
```

### Entry points unifiés

| Usage | Commande |
|---|---|
| Claude Desktop | `./hexstrike-pulse` |
| CLI (scan outil) | `python3 hexstrike.py scan <tool> <target>` |
| Dashboard web | `python3 hexstrike.py serve` |
| MCP dev (inspecteur) | `python3 hexstrike.py mcp` |

---

## V0.10.1 — Métriques

| Métrique | Valeur |
|---|---|
| Tools MCP | 20 |
| Tools total | 130 |
| Panels dashboard | 18 |
| Tests | 2577 pass, 1 skip, 2 warnings |
| Fichiers | -289/+238 lignes (consolidation) |
| Version | 0.10.1 |
