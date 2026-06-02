# Session 26 Handoff — ToolStatsStore Feed + Prefab UI Dashboard

**Date :** 2026-05-14
**Auteur :** Agent build
**Pour :** Claude

---

## Résumé

Deux livraisons majeures :

1. **ToolStatsStore feed** — la boucle d'intelligence est fermée
2. **Premier panel Prefab UI** — `pulse_app.py` opérationnel

---

## 1. ToolStatsStore Feed (feedback loop fermée)

### Ce qui a changé

```
Avant :
  run_security_tool() → _op_metrics.record()  ← métriques ops, pas stats par outil
  IDE → ToolStatsStore() privé (jamais alimenté) → blended_effectiveness() retourne baseline V6

Après :
  run_security_tool() → finalize() → get_tool_stats_store().record(tool_name, success)
       ↓
  IDE → get_tool_stats_store() (même singleton) → blended_effectiveness() retourne vrais scores terrain
       ↓
  plan_attack() génère une chain basée sur ce qui marche vraiment
```

### Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `server_core/singletons.py:60-66` | Nouveau `get_tool_stats_store()` lazy singleton |
| `mcp_core/server_setup.py:15,813-816` | Import + `finalize()` → `record(tool_name, success=...)` |
| `server_core/intelligence/intelligent_decision_engine.py:10,427` | Remplace `ToolStatsStore()` privé par `get_tool_stats_store()` |
| `tests/test_intelligent_decision_engine_core.py:6-8,855` | Patch via singleton, plus `_tool_stats` module-level |
| `tests/test_intelligent_decision_engine_phase4b.py:161-167,632` | Fixture `mock_tool_stats` → `patch.object(get_tool_stats_store(), ...)` |

### Design

- `ToolStatsStore.record(tool, success)` — thread-safe, écriture atomique disque
- `MIN_RUNS_FOR_LIVE = 5` — seuil avant de truster le live rate
- `blended_effectiveness(tool, baseline)` — retourne baseline si < 5 runs, sinon live rate
- Singleton partagé = IDE + `run_security_tool` écrivent/lisent le même store
- Persistence : fichier JSON dans `.hexstrike_data/tool_stats.json`

### Points d'attention

- `tool_stats_store.py` déjà à **100% coverage** (18 tests)
- Si un outil a < 5 runs, le baseline V6 de `tool_effectiveness` dict est utilisé inchangé
- Le `record()` est appelé dans `finalize()` **après** le cache hit check — les cache hits sont aussi comptés comme succès

---

## 2. Prefab UI Dashboard — `pulse_app.py`

### Ce qui est livré

Fichier : `pulse_app.py` (137 lignes)

```python
FastMCPApp("Pulse Dashboard")
  ├── @app.ui("pulse_dashboard")    → entry point (visible au modèle)
  └── @app.tool("get_pulse_metrics") → backend (visible à l'UI seulement)
```

### UI actuelle

- Header : badge `PULSE` + uptime + count d'outils
- Metrics row (4x `Metric`) : exécutions, taux de succès, erreurs, cache hit rate
- Recent Activity : liste des 10 derniers scans via `ForEach`

### Vérification

```bash
# Les deux tools sont bien enregistrés :
python3 -c "
import asyncio
from pulse_app import app
async def main():
    tools = await app.list_tools()
    for t in tools:
        print(t.name)
asyncio.run(main())
"
# Output: get_pulse_metrics, pulse_dashboard
```

### Test visuel

`fastmcp dev apps pulse_app.py` lance le **MCP Inspector**, pas un renderer Prefab.
Le rendu Prefab UI réel nécessite **Claude Desktop** connecté au serveur.

Le `fastmcp dev` confirme que les tools/UI sont bien enregistrés.

### requirements.txt

```txt
fastmcp[apps]>=3.1.1
prefab-ui>=0.14.0,<0.15.0
```

### Prochaines étapes possibles

1. **Ajouter SetInterval + auto-refresh** (cf. System Monitor example)
2. **Panels supplémentaires** : sparkline CPU/Mem (Histogram), cache stats, tools par catégorie (Accordion)
3. **Panel Intelligence** : diff baseline→live rate pour chaque outil après S23 feed
4. **Intégrer** dans `hexstrike_server.py` ou `hexstrike_mcp.py` via `mcp.add_provider(pulse_app)`

---

## Tests

```
2523 passed, 1 skipped (parallel)
5 pre-existing failures in serial mode (state leakage — pass in isolation)
0 régressions liées aux changements
```

### Pré-existant (inchangé)

- `test_main_block_execution` — flaky subprocess (port 8888)
- 4 tests hexstrike_server — state leakage en mode série seulement

---

## Notes importantes pour Claude

1. **ToolStatsStore est maintenant un singleton** partagé — IDE + server_setup lisent/écrivent le même store. Plus de divergence possible.
2. **`fastmcp dev apps` n'est pas un renderer visuel** — le rendu Prefab UI nécessite un client MCP compatible (Claude Desktop). Le `dev` mode valide juste l'enregistrement.
3. **prefab-ui 0.14.x** est pinné — l'API change fréquemment. Ne pas mettre à jour sans tester.
4. **Aucune intégration serveur** pour l'instant — `pulse_app.py` est standalone. L'intégration dans `hexstrike_server.py` est une prochaine étape.

---

## Versions

| Component | Version |
|-----------|---------|
| fastmcp | 3.2.4 (installé) |
| prefab-ui | 0.14.1 (pinné >=0.14.0,<0.15.0) |
| Python | 3.13.12 |
