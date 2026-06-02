# Session 43.5 — Coverage baseline + beartype pipeline

**Date** : 2026-05-18
**Branch** : `feature/prefab-dashboard`
**Tags** : `P2`, `coverage`, `beartype`, `mcp-entry`, `command-executor`, `operational-metrics`

---

## Ordre d'exécution (validé HDM)

1. **Coverage baseline** — run complet `--cov=server_core --cov=shared --cov=mcp_core`
2. **mcp_entry.py** (54%→83%) — lock file, seed cache, prewarm
3. **target_store.py** (97%→98%) — 2 missed OSError handles
4. **server_setup.py** skip (89% > 85% threshold)
5. **beartype** `_op_metrics` + `command_executor`

---

## Résultats

| Module | Avant | Après | Seuil |
|--------|-------|-------|-------|
| Global | — | **98%** (5920 stmts) | — |
| `mcp_entry.py` | 54% | **83%** | 90% |
| `target_store.py` | 97% | **98%** | 90% |
| `server_setup.py` | 89% | 89% | 85% ✅ |
| `_op_metrics` | N/A | beartype ✅ | — |
| `command_executor.py` | N/A | beartype ✅ | — |

## Tests ajoutés

**`tests/test_mcp_entry.py`** (10 tests)
- `TestSeedScanCache` (3) — seeding with/without/empty env var, 4 entries, success/failure mix
- `TestPrewarmSingletons` (3) — thread spawn, warm calls engine + store, exception handling
- `TestAcquireLock` (4) — success, stale lock removal (>30s), contention → sys.exit(1), cleanup release

**`tests/test_target_store.py`** (2 tests)
- `test_save_oserror_does_not_crash` — chmod 0o555, data preserved in-memory
- `test_save_oserror_missing_directory` — shutil.rmtree before save, no crash

## beartype

- **`OperationalMetricsStore`** (9 méthodes publiques) : `record`, `success_rate_by_tool`, `error_count_by_tool`, `timeout_count_by_tool`, `slowest_tools`, `cache_summary`, `confirmation_summary`, `cache_hits_by_tool`, `system_metrics` (static), `summary`
- **`command_executor.py`** : `execute_command()`
- PEP 585 complet (`Dict`/`List` → `dict`/`list`) sur les 2 fichiers

## Statut final

```
2626 passed, 1 skipped, 2 warnings — 0 regressions
1 flaky (TestGetPlan::test_no_target_with_explicit_none en parallèle)
```

## Gaps restants

1. `mcp_entry.py` 83% → lignes 126-135 (`_release()` closure atexit handler) — testable via `atexit._run_exitfuncs()` si souhaité
2. `server_setup.py` 89% → MCP Resources (targets://, health://, scan://...) — nouveau jeu de tests à écrire
