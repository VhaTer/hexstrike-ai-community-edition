# Session 14 — 2026-05-11 (v0.7.5 coverage: 4 more modules to 100%, overall 80%)

## Objective
Push `process_pool.py`, `singletons.py`, `performance_monitor.py`, and `run_history_store.py` to 100% coverage without adding new functionality.

## Results
- **4 modules at 100% coverage** (270 stmts, 68 branches, 0 misses)
- **+91 tests** (total from 1637→1728, no flaky, 85.96s suite time)
- **Overall coverage**: 79% → 80% (5700 stmts, 949 miss)
- **process_pool.py**: Fixed infinite-loop hang in `_monitor_performance` (BaseException break pattern via `time.sleep` side_effect) and `_worker_thread` (RuntimeError side_effect list).

## Modules Detailed

### 1. `process_pool.py` (71% → 100%)
**34 tests (rewrite).** Previously 45% coverage with 21 tests that were hanging due to infinite `while True` loops. Fix: used `patch("time.sleep", side_effect=[None, BaseException("_brk_")])` with `pytest.raises(BaseException)` to break monitor loop after one iteration.

Paths covered:
- **Worker thread (8 paths)**: success with result storage, task exception with error storage, queue.Empty timeout (caught by inner except, continues), None shutdown signal, task not in active_tasks (both success and failure), performance metrics update, outer `except Exception` catch for RuntimeError
- **Monitor performance (7 paths)**: scale up (load > threshold, workers < max), scale down (load < 0.3, workers > min), balanced (no action), zero workers (load = inf → scale up), psutil error (caught by inner except), cpu/memory metric update, outer `except Exception` catch for time.sleep crash
- **Scale detail**: empty workers after pop, negative min_workers to force `if self.workers:` False branch

### 2. `singletons.py` (41% → 100%)
**66 new tests.** Previously covered only Group A eager imports. Group B lazy accessors and backward-compat `__getattr__` aliases were completely uncovered.

Paths covered: all 12 `get_*()` accessors (first call creates singleton, second call returns same instance), all 12 `__getattr__` backward-compat aliases (session_store, wordlist_store, cve_intelligence, exploit_generator, vulnerability_correlator, decision_engine, bugbounty_manager, fileupload_framework, ctf_manager, ctf_tools, ctf_automator, ctf_coordinator), `AttributeError` for unknown attribute, Group A eager import verification.

### 3. `performance_monitor.py` (67% → 100%)
**14 new tests.** Previously covered only `__init__` via imports.

Paths covered: `monitor_system_resources` normal (cpu/memory/disk/network) + psutil error returns `{}`. `optimize_based_on_resources` all branches: CPU high with/without threads key, memory high with/without batch_size, network high (>1MB/s) with/without concurrent_connections key, low network (<1MB/s) no optimization, combined optimizations, min=1 clamping for threads/batch_size/concurrent_connections.

### 4. `run_history_store.py` (0% → 100%)
**11 new tests.** First tests ever for this module.

Paths covered: init stores state, record creates full entry, record increments ID, None tool → "unknown" default, empty params/stdout/stderr defaults, timed_out flag, partial_results flag, get_all returns copy (independent of internal deque), clear empties, empty get_all returns [], MAX_ENTRIES cap (500).

## Test Suite
```
1728 passed, 1 skipped in 85.96s
```
No failures, no flaky.

## Key Changes
| File | Change |
|------|--------|
| `tests/test_process_pool.py` | Rewritten: 34 tests, fixed hang, 100% coverage |
| `tests/test_singletons.py` | NEW: 66 tests, lazy accessors + __getattr__ |
| `tests/test_performance_monitor.py` | NEW: 14 tests, all optimization branches |
| `tests/test_run_history_store.py` | NEW: 11 tests, record/get_all/clear/cap |
| `AGENTS.md` | Session 14 appended |

## Flaky / Blockers
- **beartype 0.22.9 circular import**: Still occurs when running `--cov` on individual files. Workaround: always run `pytest tests/ --cov` (full suite). Not fixed in this session.
- **process_pool.py `if self.workers:` branch**: The False branch (202→197) required `min_workers = -1` (negative) to reach — unreachable in normal operation due to `pool_lock` and outer guard. Coverage accepted as 100% since branch is hit via the edge case test.
