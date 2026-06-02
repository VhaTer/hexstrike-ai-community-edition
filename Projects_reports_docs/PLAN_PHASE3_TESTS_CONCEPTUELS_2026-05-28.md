# Plan Phase 3 — Tests conceptuels FastMCP 3

**Date** : 2026-05-28
**Pour** : Claude (prochaine session)
**Validé par** : HDM

## 1. Réponse à Claude sur ctx

> *"dans pulse_mcp, le `ctx` est-il activement utilisé ou encore peu exploité ?"*

**Réponse courte :** `ctx` est **massivement exploité dans `server_setup.py`** (10 méthodes distinctes, 50+ appels), mais **totalement absent de `pulse_app.py`** (Prefab UI app, zéro ctx).

| `ctx.*` | server_setup.py | pulse_app.py |
|---------|:-:|:-:|
| `info` | 30+ appels | 0 |
| `report_progress` | 10 appels | 0 |
| `get_state` / `set_state` | 6 appels | 0 |
| `error` / `warning` | 5 appels | 0 |
| `session_id` | 1+ (propriété) | 0 |
| `get_prompt` / `sample` | 4 appels | 0 |
| `read_resource` | 1 appel (skills) | 0 |

**Conséquence :** `pulse_app.py` (Prefab UI) utilise son propre système — `logging` + `PrefabApp(state=dict)` + `Rx()` pour la réactivité. Les tests `ctx` ne concernent que `server_setup.py`.

---

## 2. Gaps identifiés

### Resources — 9/13 sans test (GAP CRITIQUE)

| Resource | Test existe ? |
|----------|:-:|
| `health://server` | ✅ |
| `scan://{target}/latest` | ❌ |
| `scan://{target}/{tool_name}` | ❌ |
| `scan://cache/list` | ❌ |
| `metrics://tools` | ❌ |
| `telemetry://summary` | ❌ |
| `telemetry://recent` | ❌ |
| `telemetry://tools/{tool}` | ❌ |
| `errors://statistics` | ❌ |
| `targets://` | ✅ |
| `target://{target}` | ✅ |
| `target://{target}/findings` | ✅ |
| `target://{target}/sessions` | ✅ |

### Context — 3 gaps

| Méthode | Statut |
|---------|:------:|
| `ctx.info` / `ctx.error` | ✅ testés |
| `ctx.warning` | ⚠️ mocké mais jamais asserté |
| `ctx.read_resource` | ⚠️ seulement chemin d'exception testé |
| `ctx.sample` | ⚠️ 1 seul test |

### Prompts — pas de tests d'intégration

6 prompts enregistrés, tests unitaires existent mais zéro test d'intégration.

---

## 3. Plan d'exécution

### Bloc A — Resources (priorité haute, ~12 tests)
- `test_scan_resources.py` : `scan://{target}/latest`, `scan://{target}/{tool_name}`, `scan://cache/list`
- `test_metrics_resource.py` : `metrics://tools`
- `test_telemetry_resources.py` : `telemetry://summary`, `telemetry://recent`, `telemetry://tools/{tool}`
- `test_errors_resource.py` : `errors://statistics`

### Bloc B — Prompts intégration (priorité moyenne, ~6 tests)
- Validation des messages générés par chaque prompt

### Bloc C — Context gaps (priorité basse, ~5 tests)
- Compléter `test_fastmcp3_ctx_methods.py` : `ctx.warning`, `ctx.read_resource` success, `ctx.sample`

### Bloc D — Pipeline conceptuel (priorité haute, ~6 tests)
- Risk level cohérent, next_suggested_tool pertinent, ports dédupliqués

**Total estimé : ~29 tests, ~5h30**
