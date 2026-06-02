# Session 43 — P2 scan() entry point tests + bugfix

**Date** : 2026-05-18
**Branch** : `feature/prefab-dashboard`
**Tags** : `P2`, `scan-entry-point`, `target-store`, `test-integration`, `bugfix`

---

## Exchange HDM ↔ Claude

### HDM → Claude (après break)

Étape 2 validée — 26 tests TargetStore, beartype, thread safety, fichier corrompu, 2603 passed 0 regressions. Nickel.

Sur les cibles contrôlées — on commence avec localhost, pas de container Docker pour l'instant. On ajoute testlab quand les tests de base tournent.

Go étape 3 — tests d'intégration scan() entry point avec mock tool output. 🤝

### Claude → HDM (plan mode)

Plan déposé — 10 tests, `mock_direct_tools` fixture, cache vide réaliste, 3 intensités + cache + failures + edges.

### HDM → Claude

Plan validé — mock_direct_tools propre, fixtures réutilisables, comportement naturel du cache vide c'est plus réaliste que de tout mocker.

3 remarques avant de coder :

1. **Target explicite vs scope auto-detect** — ajouter un test : scan sans target explicite doit auto-détecter depuis le scope.
2. **session_id dans les résultats** — valider que chaque run a les bonnes clés status/returncode/error.
3. **TargetStore vérifié** — les tests incluent-ils une vérification que record_scan() est appelé après scan ?

Go avec ces ajouts. 🤝

### Claude → Code (build mode)

**Bug trouvé** : `_collect_dashboard_state()` utilisait `history` ligne 1100 avant sa définition ligne 1117 → `NameError` catché silencieusement → TargetStore jamais alimenté.

**Fix** : déplacé `history = get_history(active_target)` + `active = get_active_tools()` avant le bloc TargetStore (lignes 1094-1095).

**12 tests écrits :**

| Test | Vérifie |
|------|---------|
| `test_scan_quick_basic` | 2 tools completed, structure correcte |
| `test_scan_medium` | 4 tools : nmap + whatweb + nuclei + nikto |
| `test_scan_full` | 5 tools + plan présent |
| `test_scan_cached_tool_skips_execution` | nmap en cache → "cached", whatweb exécuté |
| `test_scan_tool_failure_reported` | nmap échoue → "failed" + error |
| `test_scan_unknown_tool_skipped` | gobuster supprimé → "skipped" |
| `test_scan_invalid_intensity_defaults_to_quick` | "extreme" → fallback quick |
| `test_scan_no_target_returns_error` | cache vide + pas de target → error |
| `test_scan_auto_detects_target_from_scope` | cache prégarni → target auto-détectée |
| `test_scan_result_keys` | 7 clés top-level, status/returncode/error |
| `test_dashboard_calls_targetstore_record_scan` | spy record_scan appelé avec data |
| `test_dashboard_record_scan_skipped_when_no_data` | spy record_scan PAS appelé si vide |

### Résultats

```
62/62 pulse_app tests ✅
2613 passed, 2 flaky subprocess timeout (pré-existants), 0 regressions ✅
```

---

## Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `pulse_app.py:1094-1095` | Bugfix : `history`/`active` déplacés avant TargetStore |
| `tests/test_pulse_app.py:541-717` | Nouvelle classe `TestScanEntryPoint` (12 tests) |
| `AGENTS.md` | Session 43 ajoutée |
| `server_core/target_store.py` | beartype + PEP 585 (Session 42, appliqué) |
| `tests/test_target_store.py` | 26 tests TargetStore (Session 42, appliqué) |

---

## Prochaines étapes P2

- Coverage baseline (`pytest-cov` déjà installé)
- beartype sur reste de la data pipeline : `_op_metrics`, `command_executor`, `EnhancedCommandExecutor`
- Seuil minimal de coverage à définir
