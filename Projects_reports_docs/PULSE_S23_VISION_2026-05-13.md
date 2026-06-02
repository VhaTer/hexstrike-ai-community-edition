# Pulse S23 — Vision & Roadmap (2026-05-13)

De : Claude + Nexus
Pour : HexDevMaster
Type : Document de vision — à lire avant S23

---

## Le fil conducteur

On a eu une session de réflexion architecture aujourd'hui.
Voici ce qui a été décidé et pourquoi.

---

## 1. Fermer la boucle intelligence — ToolStatsStore

### Le problème actuel

L'IDE (`intelligent_decision_engine.py`) a tout ce qu'il faut :
- `ToolStatsStore` instancié : `_tool_stats = ToolStatsStore()`
- `blended_effectiveness(tool, baseline)` qui mixe score statique + stats live
- 14 patterns d'attaque avec scores d'efficacité par tool

Mais personne n'écrit dans `ToolStatsStore` après exécution.
Nuclei peut timeout 10 fois — l'IDE continue à lui donner 0.95 (score V6 codé en dur).
Le LLM reçoit `success_probability=0.95` — un chiffre qui ne veut rien dire terrain.

### Le fix — une ligne dans `finalize()` de `run_security_tool()`

```python
def finalize(result: Any) -> Dict[str, Any]:
    normalized = _normalize_tool_result(result)
    _telemetry["success"] = bool(normalized.get("success", False))
    _telemetry["timed_out"] = bool(normalized.get("timed_out", False))
    _telemetry["duration"] = round(time.time() - _t_start, 3)
    logger.info("[telemetry] %s", json.dumps(_telemetry))
    _op_metrics.record(_telemetry)

    # ← CETTE LIGNE FERME LA BOUCLE
    _tool_stats.record_run(
        tool_name,
        success=_telemetry["success"],
        duration=_telemetry["duration"]
    )

    return normalized
```

### Ce que ça change

```
Exécution réelle
      ↓
ToolStatsStore.record_run()
      ↓
blended_effectiveness() retourne score adaptatif
      ↓
IDE choisit les tools avec les vrais scores terrain
      ↓
plan_attack() génère une chain basée sur ce qui marche vraiment
      ↓
Claude reçoit success_probability=0.71 (32 runs, 8 timeouts sur HTB)
      ↓
Le LLM se nourrit de données réelles, pas de scores V6
```

**Le LLM devient plus intelligent parce que les données sous-jacentes sont honnêtes.**

### Vérifier la signature de ToolStatsStore

```bash
grep -n "def record_run\|def blended_effectiveness" \
  server_core/tool_stats_store.py
```

Adapter le call si la signature diffère.

---

## 2. Coverage — finir la branche avant Prefab

### État actuel

```
server_setup.py    91%  (57 miss — paths runtime-heavy)
tool_stats_store.py  ?  (jamais mesuré proprement)
```

### Objectif avant MCP Apps

Atteindre **95%+ global** — pas 100%, pas de chasse aux miss impossibles.
Les 57 miss de `server_setup.py` sont des paths runtime-heavy justifiés.
`tool_stats_store.py` mérite une passe propre.

```bash
pytest tests/ --cov --cov-report=term-missing -q \
  --ignore=tests/test_real_integration.py \
  --ignore=tests/test_server_setup_standalone.py \
  2>&1 | grep -E "TOTAL|tool_stats"
```

---

## 3. MCP Apps — Prefab UI Dashboard

### Ce qu'on a décidé

Le dashboard Pulse sera une **MCP App FastMCP 3** avec Prefab UI.
Pas de browser, pas de port 8888, pas de React.
UI native dans Claude Desktop — zéro token en lecture.

### Pourquoi c'est possible immédiatement

Toutes les données existent déjà en MCP Resources :

| Resource | Données |
|----------|---------|
| `health://server` | uptime, RAM, tools_count, cache_stats, op_metrics |
| `metrics://tools` | success rates, errors, timeouts, cache, confirmations |
| `errors://statistics` | classification erreurs par type |
| `scan://cache/list` | derniers scans session |
| `scan://{target}/latest` | dernier résultat par cible |

Zéro token consommé — lecture directe des resources.

### Les panels retenus (sketch validé par Nexus)

```
┌─────────────────────────────────────────────────────┐
│  HEXSTRIKE · PULSE  v0.8.0  healthy · 4h32m  13.7% │
├──────────┬──────────┬──────────┬───────────────────┤
│ exéc 108 │ succès94%│ cache 67%│ timeouts 3        │
├──────────┴──────────┴──────────┴───────────────────┤
│ live ops      │ performance   │ cache              │
│               │               │                    │
├───────────────┴───────────────┴────────────────────┤
│ sécurité terrain          │ rate limits            │
│ (destructive tools)       │ (profils forcés)       │
└───────────────────────────┴────────────────────────┘
```

Dev stats virées. Font mono. Thème crimson.

### Panel bonus — Intelligence (post ToolStatsStore feed)

Une fois la boucle fermée (point 1) :

```
┌─────────────────────────────────────┐
│ intelligence                        │
│ nuclei   baseline 0.95 → live 0.71  │
│ sqlmap   baseline 0.90 → live 0.88  │
│ ffuf     baseline 0.90 → live 0.62  │
│                                     │
│ IDE : adaptatif  (32 runs enregistrés)│
└─────────────────────────────────────┘
```

Ce panel rend visible que le LLM se nourrit de vraies données terrain.

### Architecture MCP App

```python
from fastmcp import FastMCP, FastMCPApp
from prefab_ui.app import PrefabApp

app = FastMCPApp("Pulse Dashboard")

@app.tool()
async def get_pulse_metrics() -> dict:
    """Lit metrics://tools — zéro token."""
    # accès direct _op_metrics.summary()
    ...

@app.ui()
async def pulse_dashboard() -> PrefabApp:
    """Ouvre le dashboard Pulse dans Claude Desktop."""
    # panels : header, métriques, live ops,
    #          performance, cache, sécurité terrain,
    #          rate limits, intelligence (post S23)
    ...
```

### Install avant de coder

```bash
source hexstrike-env/bin/activate
pip install "fastmcp[apps]"
pip show prefab-ui  # noter la version exacte
# puis pinner dans requirements.txt : prefab-ui==X.Y.Z
```

Doc : `https://gofastmcp.com/apps/overview`
Doc FastMCPApp spécifiquement : `https://gofastmcp.com/apps/interactive-apps`

---

## 4. Avis de Claude sur la CE → MCP App

### Ce qui est là et réutilisable

La CE actuelle (HTML/JS pur, SSE temps réel, 6 panneaux, dashboard crimson V6)
a résolu les bons problèmes — SSE stream, layout, thème.
Mais elle vit sur port 8888, nécessite un browser séparé,
et n'est pas intégrée dans le flow Claude Desktop.

### Ce que MCP Apps change fondamentalement

| CE actuelle | MCP App Prefab |
|-------------|----------------|
| Browser séparé | Native Claude Desktop |
| Port 8888 | Aucun port |
| SSE polling | Resources MCP directes |
| HTML/JS custom | Python + Prefab composants |
| Démarrage serveur requis | Intégré au serveur MCP |
| Zéro token | Zéro token ✓ |

### Recommandation

Ne pas porter la CE vers Prefab 1:1.
Partir du sketch validé (6 panels terrain) et construire proprement.
Le thème crimson + font mono sont les seuls éléments à préserver.
Le reste est à reconstruire avec les primitives Prefab — plus simple, plus maintenable.

La CE reste en place comme fallback browser si besoin.
Mais le dashboard Prefab est la direction principale.

---

## Ordre d'exécution S23

1. `ToolStatsStore.record_run()` dans `finalize()` — 10 minutes
2. Tests pour la nouvelle ligne + `tool_stats_store.py` coverage
3. `pip install "fastmcp[apps]"` + pin prefab-ui
4. Lire la doc FastMCPApp
5. Premier dashboard Prefab — panels header + métriques + live ops
6. Itérer sur les panels restants

**Ne pas tout faire en une session.**
S23 = ToolStatsStore feed + premier panel Prefab fonctionnel.
S24 = panels complets + panel intelligence.

---

## Handoff attendu

`Projects_reports_docs/AGENT_HANDOFF_2026-05-13_S23.md` avec :
- ToolStatsStore feed : implémenté + testé
- prefab-ui version pinnée
- premier panel Prefab : screenshot ou description rendu
- coverage avant/après
- commit hash
