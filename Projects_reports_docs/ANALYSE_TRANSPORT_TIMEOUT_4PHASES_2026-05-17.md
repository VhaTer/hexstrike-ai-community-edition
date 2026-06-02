# Analyse transport & timeout — Plan 4 phases corrigé

**Date :** 2026-05-17
**Auteur :** HexDevMaster
**Pour :** HexReaper + Claude (Sonnet 4.6)
**Référence :** CORRECTION_ARCHITECTURE_TRANSPORT_RPI_2026-05-17.md

---

## Rappel des corrections architecture

| Point | Avant | Après |
|---|---|---|
| Claude Desktop ↔ HTTP | Pensé possible | **Stdio uniquement** — pas de `url` dans config |
| RPI | Pensé serveur MCP potentiel | **Cible pure** sur réseau local |
| HTTP transport | Pensé prioritaire pour timeout | **Utile seulement pour Cline/Continue/clients tiers** |

---

## 1. Le vrai problème : timeout 300s stdio

### Chaîne d'exécution actuelle

```
MCP tool call (Claude Desktop)
  → hexstrike_mcp.py (stdio)
    → run_security_tool()     ← async, await synchrone
      → loop.run_in_executor() ← bloque dans un thread
        → exec_func(tool_key, params)
          → command_executor.execute()
            → EnhancedCommandExecutor()
              → subprocess.Popen() + process.wait(timeout=300)
                └──→ retourne résultat après N secondes
```

**Problème :** `run_security_tool()` await le résultat du scan. Si le scan dépasse ~300s, Claude Desktop considère le tool comme mort et affiche une erreur, même si le scan a réussi en backend.

### Ce qui existe déjà (mais pas utilisé pour ça)

```
EnhancedProcessManager.execute_command_async()
  → submit_task(task_id, ...)    ← retourne immédiatement
  → get_task_result(task_id)     ← polling
```

`run_security_tool()` n'utilise PAS `EnhancedProcessManager`. Il passe directement par `asyncio.run_in_executor()` → blocking.

---

## 2. Plan 4 phases — version corrigée

### Phase 1 — Unifier les entry points (30 min)
**Pour :** Cline/Continue/clients HTTP, pas Claude Desktop

- Ajouter `--transport {stdio,http}` + `--port` à `hexstrike_mcp.py`
- Ajouter `pulse_app` comme provider à `hexstrike_server.py` (manquant : pas de tools dashboard en HTTP)
- Unifier `_acquire_lock()`, `_prewarm_singletons()`, `_seed_scan_cache()` entre les deux scripts
- Pas de changement pour Claude Desktop (reste stdio)

**Fichiers :** `mcp_core/mcp_entry.py`, `mcp_core/args.py`, `hexstrike_server.py`

### Phase 2 — SUPPRIMÉE
**Raison :** Claude Desktop ne supporte pas HTTP. La config `url` n'existe pas pour les MCP servers locaux.

### Phase 3 — Fix timeout 300s par execution async (PRIORITAIRE)

**Principe :** Rendre les tools MCP non-bloquants pour que Claude Desktop reçoive une réponse immédiate, pendant que le scan continue en background.

```
Avant :
  run_security_tool() → await scan → 300s → retour résultat → Claude voit TOUT

Après (async mode) :
  run_async_tool(tool, target, params)
    → lance scan via EnhancedProcessManager.execute_command_async()
    → retourne IMMÉDIATEMENT {scan_id, status: "started"}
    → Claude Desktop voit réponse < 100ms → pas de timeout
  
  get_scan_status(scan_id)
    → retourne {status, progress, result (si fini)}
    → Claude/User peut poller ou vérifier plus tard

  get_history(target)
    → scan terminé apparaît automatiquement
```

**Ce qui change :**

| Avant | Après |
|---|---|
| `run_security_tool()` await synchrone | Nouveau tool `run_async_tool()` launch + return |
| Résultat dans la réponse MCP | Résultat dans `_scan_cache`, pollable via `get_scan_status()` |
| Timeout 300s = scan cassé | Timeout 300s = juste la réponse "scan démarré" (<100ms) |
| Dashboard snapshot périmé | Dashboard peut montrer scan en cours + résultats frais |

**Architecture détaillée :**

```
@app.tool()
def run_async_tool(tool: str, target: str, params: str = "", ) -> dict:
    """Lance un outil en background. Retourne immédiatement un scan_id."""
    # 1. Construire la commande
    # 2. Lancer via EnhancedProcessManager.execute_command_async()
    # 3. Stocker mapping {scan_id: {tool, target, task_id, start_time}}
    # 4. Retourner scan_id + status immédiatement

@app.tool()  
def get_scan_status(scan_id: str) -> dict:
    """Statut d'un scan lancé via run_async_tool."""
    # 1. Récupérer task_id via mapping
    # 2. EnhancedProcessManager.get_task_result(task_id)
    # 3. Retourner status + result (si complété) + progress

# Évolution du dashboard :
# → Nouvel onglet "Running Scans" (déjà partiellement via Active Tools)
# → get_history() montre les scans async complétés
```

**Considérations threading :**
- `EnhancedProcessManager` utilise `ProcessPool` (thread pool) — thread-safe
- `_op_metrics.record()` est thread-safe (lock)
- `_scan_cache.set()` est thread-safe (lock)
- Le scan_id mapping doit être thread-safe (dict + Lock)

**Tests :**
- Nouveau test : lancer `run_async_tool()` → vérifier `scan_id` non-null
- Nouveau test : `get_scan_status(scan_id)` → vérifier `status` = "running" puis "completed"
- Pas de régression attendue

**Fichiers :** `pulse_app.py`, `mcp_core/server_setup.py`

### Phase 4 — Dashboard temps réel

**Pour clients HTTP (Cline, Continue, navigateur) :**
- `/web-dashboard/stream` (SSE) existe déjà dans `hexstrike_server.py`
- Déjà fonctionnel : stream JSON toutes les 2s avec données fraîches
- Après Phase 3, les scans async apparaîtront dans le stream

**Pour Claude Desktop :**
- Dashboard inline HTML généré par Claude depuis les tools MCP
- Après Phase 3, les données sont plus fraîches car les tools ne bloquent pas
- Possibilité d'ajouter un tool `get_live_dashboard()` qui retourne tout l'état en un appel
- Connecter le dashboard PrefabApp à `get_scan_status()` pour montrer les scans en cours

---

## 3. Priorités roadmap

```
S36.5   Phase 3 — Async tools     ⬅ PRIORITAIRE (fix timeout 300s)
S37     Phase 1 — Entry points     ⬅ Pour Cline/Continue
S38     Phase 4 — Dashboard live   ⬅ Après Phase 3 stable
RPI     Test end-to-end            ⬅ Dépend de Phase 3 pour scans > 300s
```

La Phase 3 débloque tout le reste :
- Claude Desktop ne perd plus les scans longs (sqlmap, nuclei, nikto)
- Le dashboard peut montrer des scans en temps réel
- Le RPI vuln lab peut être testé sans timeout arbitraire

---

*HexDevMaster — HexStrike team — 2026-05-17*
