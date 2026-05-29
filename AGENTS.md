# HexStrike AI-PULSE — Agent Guide

## Setup

```bash
source hexstrike-env/bin/activate
python3 hexstrike_mcp.py --debug          # stdio MCP server
python3 hexstrike_server.py               # HTTP server
fastmcp dev apps pulse_app.py             # Prefab UI validation
```

## Test commands

```bash
python -m pytest tests/ -m "not slow" -n auto -q
python -m pytest tests/ -m slow -n auto -q --tb=short     # slow ones only (benchmarks, concurrent, real integration)
python -m pytest tests/test_pulse_app.py -v -q --tb=short
python -m pytest tests/test_server_setup_unit.py -k TestScanBackground -q --tb=short  # scan_background tests (10, ~25s)
python -m pytest tests/test_cache_benchmarks.py -q --tb=short --benchmark-only   # benchmarks sans xdist
python -m pytest tests/test_cache_concurrent.py -n 4 -q --tb=short               # concurrent load tests (9 tests, ~48s)
python -m pytest tests/test_rpi.py -m rpi -v --tb=short                           # 14 RPI real target tests (~8min, needs RPI on 192.168.1.165)
```

## Key conventions

- **Branch**: `feature/prefab-dashboard` — single branch for all Prefab dashboard work
- **Merge**: only when user judges stable. No auto-merge.
- **Prefab UI**: `@app.tool()` for backend, `@app.ui()` for entry point. State set once at creation.
- **Scope**: active target auto-detected from most recent `_scan_cache` entry.
- **Data sources**: `_op_metrics` for operational stats, `_scan_cache` for scan results, `get_tool_stats_store()` for tool effectiveness.
- **State keys**: flat dict in `PrefabApp(state={...})`, accessed via `Rx("key").then(lambda v: ..., default)`.
- **Search transform**: `RegexSearchTransform` (zero overhead, pattern matching). `always_visible` = 4 essential tools (`scan`, `get_live_dashboard`, `scan_background`, `pulse_dashboard`) + 2 meta-tools auto-injected (`search_tools`, `call_tool`). All other tools discovered via `search_tools()`. `max_results=10`, markdown serializer.
- **Token budget**: ~6 tools fixed in `list_tools()`. No pentest tools pre-loaded. Purpose-driven consumption via search only when needed.
- **Exception handling**: bare `except Exception: pass` interdit. Chaque `except` doit logguer (au minimum `logger.debug()`). Les chemins best-effort doivent répondre à "si ça fail, est-ce que je veux le savoir ?".

## Session 27 — 2026-05-14 (cleanup + Prefab 6 panels)

- Removed 5 orphan deps (selenium, aiohttp, mitmproxy, beautifulsoup4, webdriver-manager)
- Full Prefab dashboard with 6 panels (Status Bar, Resource Gauges, Cache+Errors, Execution Activity, Tools by Category, Intelligence)
- 4 backend tools + 1 UI entry point
- Branch `feature/prefab-dashboard` merged to master, tagged v0.8.0
- 2524 passed, 1 skipped, 2 warnings

## Session 28 — 2026-05-14 (v0.8.0 Prefab v2 — Header + Scope)

- New workflow: feature branch -> test -> merge when stable
- `_op_metrics.system_metrics()` now returns `memory_total_gb`
- `get_overview()` tool: version, uptime, RAM, tools count, server status
- `get_scope()` tool: auto-detects active target from most recent `_scan_cache` entry, detects type (ip/domain/url), shows tools used + timing
- **Header panel**: one-liner `PULSE v0.8.0 • up Xh Ym • RAM X/Y GB • N tools • ✓ healthy`
- **Scope panel**: Card with target, type badge, tool badges, summary line
- Legacy panels kept: System Resources, Recent Activity, Intelligence DataTable
- 5 dev panels removed: Cache+Errors, Tools by Category, old get_pulse_metrics
- 26 tests, 2550 total, 0 regressions

## Session 29 — 2026-05-15 (v0.8.0 Prefab v2 — Full dashboard with Plan IDE + Active Tools + History)

- Surface + Findings panels (from v0.8.0 Session B plan)
- `get_surface(target)` tool: parses nmap stdout for open ports/services, whatweb for tech detection, computes risk level (high >5 ports, medium >2 ports). Falls back to scope.
- `get_findings(target)` tool: parses nuclei terminal output (`[severity] [id] ...`) sorted by severity. Parses nikto (`+ /: finding`) for info issues.
- `_cache_for_target(target)` helper: filters `_scan_cache` by target across all sessions.
- `get_plan(target, objective)` tool: attack chain from `IntelligentDecisionEngine`, returns steps with prob/ETA.
- `get_active_tools()` tool: running processes from `EnhancedProcessManager`.
- `get_history(target, limit)` tool: scope-filtered scan history, replaces `get_recent_scans()`.
- `get_pulse_data(target)` tool: aggregates all 10 tools into a single JSON response.
- **Plan IDE panel**: DataTable with #/Tool/Outcome/Prob/ETA.
- **Active Tools panel**: Card with Metrics (Processes/Workers/Queued) + summary.
- **Historique panel**: DataTable (Tool/Target/When/Status/Time) replaces Recent Activity.
- Binary name fixes (7 tools underscore→hyphen), Pulse CLI removed, CTF made effective, orphaned files cleaned.
- 53 tests, 2580 total, 0 regressions.

## Session 30 — 2026-05-15 (Lazy init fixes — ParameterOptimizer, CTF automator, os.makedirs, pre-warming)

- `intelligent_decision_engine.py`: `ParameterOptimizer()` moved from module-level (eager) to `_get_parameter_optimizer()` lazy method on class. Saves ~50ms on first `get_decision_engine()` call.
- `ctf/automator.py`: removed module-level `CTFWorkflowManager()` and `CTFToolManager()` instances (duplicate singletons). Replaced with `self._manager` / `self._tools` lazy properties that delegate to `get_ctf_manager()` / `get_ctf_tools()` from singletons.
- `config_core.py`: added `resolve_data_dir()` + `ensure_data_dir()` (thread-safe, idempotent). `tool_stats_store.py`, `session_store.py`, `wordlist_store.py` now use it — eliminates 2 redundant `os.makedirs()` syscalls.
- `mcp_entry.py`: added background `_prewarm_singletons()` thread that pre-initializes `get_decision_engine()` + `get_tool_stats_store()` after server start. First dashboard call no longer blocks ~100-350ms.
- `pulse_app.py`: `get_pulse_data()` converted to `async def` with `asyncio.gather()` for 6 heavy sub-calls in parallel. Drops latency from ~300ms sequential to ~100ms parallel (limited by slowest sub-call).
- `_safe_call()` helper: wraps each sync function in `asyncio.to_thread()` with graceful fallback on exception.
- 2580 passed, 1 skipped, 2 warnings — 0 regressions.

## Session 31 — 2026-05-16 (Design analysis + Prefab docs + Red team audit)

- Prefab UI docs explorées via ctx7 : découverte de `Icon` (Lucide), `Sparkline`, `Grid`, `Tooltip`, `Progress`, `Carousel`, `CardHeader/Title/Footer` — composants existant dans le même renderer que Badge/Text/DataTable, jamais utilisés
- 16 data sources auditées : 30% seulement sont affichées sur le dashboard
- Data cachées clefs : `RateLimitDetector` (profile/target), `IntelligentErrorHandler` (error types, tool alternatives), `slowest_tools()`, `cache.stats()`, `ProcessPool.performance_metrics`, `SessionStore`, network I/O, confirmation_summary()
- Red teamer evaluation : dashboard bon pour "quoi maintenant ?" mais pas pour "quoi faire différemment ?". Top 3 manquants = détection, fiabilité, efficacité
- Header repensé proposé : icônes Lucide + Sparkline CPU + Tooltip → remplacer texte "RAM X/Y GB"
- 7 nouveaux panels identifiés (Detection Status, Errors & Failures prioritaires)
- Live test Claude Desktop S30 validé : nmap 10.74s, whatweb 13.2s, dashboard inline HTML
- Problèmes connus : double instance MCP (lock file à faire), cache seed non consommé (à diagnostiquer)
- `CLAUDE_FEEDBACK_S31_DEBRIEF_2026-05-16.md` créé pour le retour de Claude
- 2580 passed, 1 skipped, 2 warnings — 0 regressions.

## Session 32 — 2026-05-17 (Header redesign — Icon + Sparkline + Tooltip)

- Header texte remplacé par icônes Lucide (`cpu`, `hard-drive`, `database`) + Progress bars
- `Tooltip` au survol de chaque stat (CPU/RAM/Disk)
- `Sparkline` CPU en dessous du header (fill, smooth curve, 16px)
- `Div` comme flex spacer entre les stats et le nom de version
- `cpu_history` extrait de `ResourceMonitor.usage_history` pour alimenter le sparkline
- Nouveaux Rx : `CPU_PCT`, `RAM_DETAIL`, `DISK_PCT`, `CPU_SPARK`
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- Brief : `Projects_reports_docs/BRIEF_S32_HEADER_REDESIGN_2026-05-17.md`

## Session 33 — 2026-05-17 (Errors & Failures panel + merge)

- Header bugfix: `with` keyword on Tooltip/Row context managers, `Span→Div`, new state keys
- `get_errors_and_failures()` tool: aggregates `IntelligentErrorHandler.get_error_statistics()` + `_op_metrics.error_count_by_tool()`, `timeout_count_by_tool()`, `slowest_tools()`
- `error_handler` imported from `server_core.singletons`
- Panel placed between Rate Limit & Intelligence (4 sub-tables)
- Commit `ad7cea0` on master, 13 ahead of origin
- 2576 passed, 1 skipped, 2 warnings — 0 regressions
- Docs: `CLAUDE_S33_STATE_OF_PLAY_2026-05-17.md`, `CLAUDE_FEEDBACK_S33_DEBRIEF_2026-05-17.md`

## Session 34 — 2026-05-17 (Cache seed fix + Panels Batch 1)

- Cache seed fix: removed `:{timestamp}` from seed key format (`seed:{tool}:{target}`), added fallback `_scan_cache.get(f"seed:{tool_name}:{target}")` in `run_security_tool()`. Seeds now serve as real cache hits.
- `_op_metrics.cache_hits_by_tool()` — new method exposing per-tool cache hit counts
- Tool Performance panel: `get_tool_performance()` — success rate (worst-first) + timeouts
- Cache Status panel: `get_cache_status()` — hits/misses/ratio + size/utilization + per-tool cache hits
- System Trends panel: `get_system_trends()` — CPU/memory 10-step avg + sparklines (30-point history)
- 14 panels total on dashboard
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- Brief: `Projects_reports_docs/BRIEF_S34_CACHE_FIX_PANELS_BATCH1_2026-05-17.md`

## Session 35 — 2026-05-17 (Cold start fix + Panels Batch 2 — Sessions, Confirmations, Network I/O)

- Tool descriptions enrichies pour get_overview, get_surface, get_findings, get_plan — chaque description passe de ~5 à ~40-60 mots avec workflow hints, return structure, exemples. Couche 1 plug-and-play.
- Sessions panel: `get_sessions()` — SessionStore active + completed sessions
- Confirmations panel: `get_confirmations()` — accepted/denied/skipped
- Network I/O panel: `get_network_io()` — bytes sent/recv from ResourceMonitor
- 17 panels total on dashboard
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- Brief: `Projects_reports_docs/BRIEF_S35_PANELS_BATCH2_TRANSPORT_2026-05-17.md`

## Session 36 — 2026-05-17 (Transport audit + Couche 1 typed tools + v0.10.0 push)

- Transport audit: no `--transport` arg, stdio only. FastMCP supports HTTP/SSE but not wired.
- `_TOOL_COUCHE1` dict: workflow hints + examples for nmap, whatweb, sqlmap, gobuster, nuclei, nikto
- `_build_typed_tool_doc()` generates richer template for all 130+ tools (Workflow/Args/Returns/Example)
- `server_health` description enriched directly
- Tag v0.10.0 + push origin (16 commits, 2 tags)
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- Brief: `Projects_reports_docs/BRIEF_S36_TRANSPORT_V0100_2026-05-17.md`
- Cycle S31-S36 terminé — prochain milestone : RPI vuln lab

## Session 36.5 — 2026-05-17 (Async tools + Async Scans dashboard panel)

- Architecture correction: Claude Desktop = stdio only, no HTTP/SSE transport available
- Phase 3 async tools implemented: `run_async_tool(tool, target, params)` launches background thread, returns `scan_id` immediately (2s)
- `get_scan_status(scan_id)` polls status/results (0/0s, no timeout risk)
- Background thread calls `exec_func(tool, target, **params)` directly via `get_direct_tools()` cache
- Cache mechanism: `_async_scans` dict protected by `threading.Lock`, tracks start/end/duration/status/output
- `mcp_core/server_setup.py`: added `_DIRECT_TOOLS_CACHE` + `get_direct_tools()` function, populated during `setup_mcp_server_standalone()` for module-level access
- Async Scans panel added to dashboard (2x DataTable: running + completed, with Summary line)
- 18 panels total on dashboard
- 2577 passed, 1 skipped, 2 warnings — 0 regressions

## Session 37 — 2026-05-17 (Entry points unifiés + Dashboard Live)

- `_collect_dashboard_state(target)` — extract data collection from `pulse_dashboard()` into shared function. UI and tool now share the same data pipeline.
- `get_live_dashboard(target)` — Phase 4 deliverable: single MCP tool returning ALL 18 dashboard panels in one JSON (~100ms). Replaces 15+ individual get_* calls.
- `scan(target, intensity, objective)` — Phase 1 deliverable: unified scan entry point. Intensity levels: quick (2 tools), medium (4 tools), full (5 tools + planning). Respects cache — cached tools skipped.
- `TOOLS_BY_INTENSITY` — module-level mapping of intensity → tool lists
- 20 tools total on dashboard (2 new)
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- Brief: `Projects_reports_docs/BRIEF_S37_ENTRY_POINTS_DASHBOARD_LIVE_2026-05-17.md`

## Session 38 — 2026-05-17 (Consolidation — tool_routes + launcher script)

- `mcp_core/tool_routes.py`: shared `TOOL_ROUTES` dict — single source of truth for all 130+ tool→binary mappings. Both `hexstrike.py` and `server_setup.py` import from it.
- `hexstrike.py`: replaced duplicated `DIRECT_ROUTES` (130 lines) with `from mcp_core.tool_routes import TOOL_ROUTES`. VERSION bumped to 0.10.1.
- `mcp_core/server_setup.py`: `DIRECT_TOOLS` dict auto-built from `TOOL_ROUTES` via `_exec_by_name` dispatch table. Removed 150 lines of duplication.
- `hexstrike-pulse`: launcher script — auto-activates venv + cleans stale lock file (>30s). Single path for Claude Desktop config.
- `mcp_core/mcp_entry.py`: lock file now checks TTL (30s) — stale locks from crashes are auto-removed.
- Entry points unifiés: `hexstrike-pulse` (Claude Desktop) / `hexstrike.py` (CLI) / `hexstrike_server.py` (HTTP dashboard). Zéro confusion.
- 2576 passed, 1 skipped, 2 warnings — 0 regressions (1 flaky CLI test in parallel)

## Session 39 — 2026-05-17 (Doc header + MCP Apps discovery)

- `hexstrike-pulse`: full English header — prerequisites (venv, pip, validate), launch, entry points, Claude Desktop config JSON example
- `hexstrike.py`: fixed indent on ctf subcommand in docstring
- MCP Apps standard confirmed: FastMCP 3.2.4 `@app.ui()` uses `_meta.ui.resourceUri` + `PREFAB_RENDERER_URI` → compatible with Claude Desktop natively
- `prefab_ui` 0.19.1 installed (requirement pins <0.15.0) — potential version mismatch
- Live tests: `get_live_dashboard()` 17 keys ✅, `scan()` ✅, `get_overview()` ✅
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- Report: `Projects_reports_docs/CLAUDE_REPORT_MCP_APPS_2026-05-17.md`
- Commits: `ce2e84d` + `11d5afe` pushés sur origin/master

## Session 40 — 2026-05-17 (Downgrade prefab-ui 0.19.1→0.14.1)

- prefab-ui downgraded 0.19.1 → 0.14.1 (API incompatible)
- 50/50 pulse_app tests, 2577 full suite, 0 regressions
- Blocker : dashboard Prefab ne rendait pas dans Claude Desktop (résolu)

## Session 41 — 2026-05-17/18 (Consolidation — 4 coupes + bugs + TargetStore)

- Dashboard 18→10 sections : System Trends, Sessions, Confirmations, Network I/O supprimés
- Header : CPU + MEM sparklines côte à côte
- Rate Limit : ligne unique compacte (inline Python, zéro leak template)
- Tool Health : Errors & Failures + Performance fusionnés en 1 table
- Badge rouge : `error_variant` basé sur `recent_errors` → vert en idle
- Footer : lit `tool_stats.json` via `get_tool_stats_store().get_all_stats()` → vrai total (1621 runs)
- Launcher stdout : isolation fd 3 (WSL boot messages → stderr, JSON-RPC → stdout propre)
- `nonexistent_tool_xyz` filtré de Intelligence
- **TargetStore** : nouveau `server_core/target_store.py` — findings persistants par cible
  - MCP Resources : `targets://`, `target://{target}`, `target://{target}/findings`, `target://{target}/sessions`
  - Auto-enregistrement dans `_collect_dashboard_state()`
  - Scope bar enrichi : "3 ports · 5 vulns · 8 scans"
- Config Claude Desktop alignée sur `hexstrike-pulse` via launcher
- 2577 passed, 1 skipped, 2 warnings — 0 regressions
- P2 plan ready : integration tests + coverage + beartype sur data pipeline

## Session 42 — 2026-05-18 (P2 — beartype TargetStore + 26 tests)

- `@beartype` decorator ajouté aux 4 méthodes publiques de `TargetStore`
- Type hints migrées `Dict`/`List` → `dict`/`list` natifs (PEP 585, élimine les 10 warnings beartype)
- `get_target()` injecte désormais la clé `"target"` dans le dict retourné
- 26 nouveaux tests : construction, record_scan (ports/techs/services/vulns/dedup), add_finding (4 types + edge cases), get_all_targets, persistence (survival + corrupted file + non-dict), thread safety (concurrent record_scan + add_finding)
- 2603 passed, 1 skipped, 2 warnings — 0 regressions

## Session 43 — 2026-05-18 (P2 — scan() entry point tests + bugfix)

- Bugfix `_collect_dashboard_state()` : `history` utilisée avant sa définition (NameError silencieux → TargetStore jamais alimenté). Déplacé `history = get_history()` avant le bloc TargetStore.
- 12 nouveaux tests `TestScanEntryPoint` : 3 intensités (quick/medium/full), cache hit, tool failure, unknown tool, invalid intensity, no target, auto-detect scope, result structure, TargetStore record_scan spy (appelé + skip si vide)
- 2613 passed, 2 flaky subprocess timeout (pré-existants, même que session 38), 0 regressions
- Prochain : P2 suite — coverage → beartype sur pipeline data restante

## Session 43.5 — 2026-05-18 (P2 — Coverage baseline + beartype pipeline)

- Coverage baseline : **98% global** (5920 stmts, 124 missed)
- `mcp_entry.py` 54% → **83%** (10 tests : seed cache, lock, prewarm)
- `target_store.py` 97% → **98%** (2 tests : OSError save failure, missing directory)
- `server_setup.py` **89%** → déjà au-dessus du seuil 85%
- `@beartype` ajouté aux 10 méthodes publiques de `OperationalMetricsStore` + PEP 585
- `@beartype` ajouté à `execute_command()` + PEP 585 dans `command_executor.py`
- 2626 passed, 1 flaky (parallélisation), 0 regressions

## Session 44 — 2026-05-18 (Step 3 — TargetStore alimenté par scan())

- `scan()` entry point appelle désormais `TargetStore.record_scan()` après exécution des outils
- TargetStore est alimenté même sans passer par le dashboard
- 2 nouveaux tests : spy `record_scan` appelé (avec cache contenant des ports) + skip si aucune donnée
- 2629 passed, 1 skipped, 2 warnings — 0 regressions

## Session 44 — 2026-05-18 (P2 cleanup — 3 bugs fermés)

- **Bug #1 fix** : `"target_profile"` dupliqué avec indentation wrong dans state PrefabApp. Supprimé ligne 1640.
- **Bug #3 fix** : `@beartype` + PEP 585 ajouté à `EnhancedCommandExecutor.execute()`.
- **Bug #4 fix** : `_release_lock()` extraite comme fonction module-level (était closure dans `_acquire_lock`). Couverture 83% → 97%. 3 tests : cleanup normal, already-unlinked, None handle.
- 2630 passed, 1 flaky (parallélisation pré-existant), 0 regressions.

## Session 46 — 2026-05-19 (Dead state + Rx cleanup — 2 passes)

Pass 1:
- **17 Rx declarations supprimées** : UPTIME, RAM, STATUS, STATUS_VARIANT, SURFACE_TARGET/PORTS/TECHS, PORTS_COUNT, RECENT_SCANS, ERR_TOOL/TIMEOUTS/SLOWEST/TYPE/RECENT, CACHE_HIT_RATE, PERF_TIMEOUTS/SUM
- **16 alias `rx_*`** associés supprimés
- **34 state keys mortes** retirées de PrefabApp(state={...})
- **3 assignations locales mortes** dans `pulse_dashboard()` : `sys`, `rl_events_table`, `target_profile`

Pass 2 :
- **18 Rx redondantes** (inline dans le template déjà) : ACTIVE_PROCS/WORK/QUEUE, CACHE_HITS/MISS/RATIO/SIZE/MAX/UTIL/TOOL, AS_RUNNING/COMPLETE, PLAN_STEPS, PERF_DATA, FINDINGS, HISTORY, INTELLIGENCE, CACHE_HIT_RATE
- **16 alias `rx_*`** supplémentaires
- **2 locals mortes** : `ops`, `cache_hit_display`
- **3 clés retour mortes** de `_collect_dashboard_state()` : `sys`, `ops`, `rl_events_table`
- **2 blocs de calcul morts** : `sys = ...`, `rl_events_table = [...]`

State dict : 85→49 (−42%), Rx : 64→30 (−53%)
2631 passed, 1 skipped, 2 warnings — 0 regressions

## Session 47 — 2026-05-19 (MCP TargetStore Resources tests)

- 8 tests for 4 TargetStore MCP resources : `targets://`, `target://{target}`, `target://{target}/findings`, `target://{target}/sessions`
- Couverture : listing, target trouvé/non trouvé, findings, sessions, cache séparé via `tmp_path` + `patch(get_target_store)`
- 2639 passed, 1 skipped, 2 warnings — 0 regressions

## Session 49 — 2026-05-19 (Couche 2 + 3 — instructions + next_suggested_tool + dashboard restaured)

### Couche 2 — System prompt (`instructions=`)

- `mcp_core/instructions.py` créé — 80 lignes : entry points, workflow order, prompts, resources, naming, cache, destructive tools
- `mcp_core/server_setup.py:741` — `FastMCP(..., instructions=os.environ.get("HEXSTRIKE_INSTRUCTIONS", INSTRUCTIONS))`
- Override possible via `HEXSTRIKE_INSTRUCTIONS` env var
- Note : instructions visibles dans Claude Code (terminal) mais silencieusement ignorées par Claude Desktop (bug Anthropic connu)
- `import os` ajouté à `server_setup.py`

### Couche 3 — `next_suggested_tool` dans les réponses

**Exec tools (130+) :** `_suggest_next_tool(tool_name, output, target)` dans `server_setup.py`, injectée dans `finalize()` au moment du `_normalize_tool_result()`. Règles basées sur output parsing :
- nmap port 80/443 → whatweb, port 445 → smbmap, port 22 → hydra, DB ports → sqlmap
- whatweb → WordPress → wpscan, Joomla → joomscan, Drupal → nuclei, générique → gobuster
- nuclei/nikto → SQLi → sqlmap, XSS → dalfox, SSL → testssl, SMB/EternalBlue → metasploit
- Autres catégories : gobuster→nuclei, hydra→metasploit, smbmap→hydra/metasploit

**App tools (pulse_app) :** `_suggest_next_from_context(surface, findings)` dans `pulse_app.py`. Règles basées sur données structurées (ports, services, technologies, sévérité des findings) :
- `scan()` → `next_suggested_tool` dans le return
- `get_surface()` → `next_suggested_tool` dans le return
- `_collect_dashboard_state()` → `next_suggested_tool` dans le pipeline
- `get_live_dashboard()` → `next_suggested_tool` dans le return

### Dashboard restauré (suite à erreur S48)

- `pulse_dashboard()` (UI entry point) + template Prefab + Rx + state restauré via `git checkout HEAD`
- Correctifs S45 ré-appliqués : URL routing (`_TOOLS_NEED_URL`, `_TOOLS_NEED_URL_AS_TARGET`), cache write-back, TargetStore record_scan
- `requirements.txt` : `prefab-ui>=0.14.0,<0.15.0` restauré
- 2639 passed, 1 skipped, 2 warnings — 0 regressions

### Pre-RPI items (Session 50)

- **HTTP transport** : `--transport {stdio,http}` ajouté à `args.py`, `run_mcp()` conditionnel dans `mcp_entry.py`. `hexstrike_mcp.py --transport http --host 0.0.0.0 --port 8888` fonctionne maintenant.
- **Dead code nettoyé** : `_SKILL_SUPPORT_FILES` réduit de 3 à 1 fichier (`REFERENCE.md` seulement — CHECKLIST.md/EXAMPLES.md n'existaient pas).
- **Flaky test parallèle fixé** : `TestGetPlan::test_no_target_with_explicit_none` patch maintenant `_scan_cache` pour éviter les interférences entre tests.
- 2639 passed, 1 skipped, 2 warnings — 0 regressions

## Session 52 — 2026-05-20 (Bug fix -sn + Lock file fcntl+PID combiné + Skills template)

- **Bug fix ParameterOptimizer** : `_apply_profile()` appendait `-sS -sV` de `additional_args` même quand utilisateur set `scan_type="-sn"`. Fix : quand `scan_type` dans `caller_keys`, les flags `-s[A-Za-z]+` sont stripés du `additional_args` du profil avant append. `nmap -sn -T4 --max-retries 2` au lieu de `nmap -sn -sS -sV -T4 --max-retries 2` → ✅ 11 hosts up.
- **Lock file fcntl + PID combiné** : `_acquire_lock()` lit d'abord le PID du lock file, vérifie `/proc/<PID>` pour détection d'instance vivante (plus rapide et plus fiable que fcntl seul). `fcntl.flock()` gardé comme filet de sécurité (auto-cleanup noyau sur crash). PID écrit dans le fichier pour diagnostic. TTL réduit 30s→5s (safety net). `_release_lock()` truncate le fichier à 0 (marqueur clean shutdown) avant unlock+unlink.
- **Launcher `hexstrike-pulse`** : logique de nettoyage lock supprimée (`_acquire_lock` gère tout, évite race condition).
- **SkillsDirectoryProvider** : `supporting_files` "resources" → "template" (REFERENCE.md caché de list_resources, 39→26 resources, −33%).
- **beartype** ajouté à `requirements.txt` (était absent, brisait l'intégrité de build).
- **Flaky fix** : `test_full_mcp_startup_flow` mock `_acquire_lock` pour contention parallèle.
- **17 tests lock test_mcp_entry** (4 nouveaux : PID alive/dead/corrupt/empty/no file).
- **Rapport** : `Projects_reports_docs/CLAUDE_S52_BRIEF_2026-05-20.md`
- **Commit** : `e60d02a` (7 files, +132/−30) + `05321a9` (S49-S51 rattrapé, 3 files, +16/−9)
- 2660 passed, 1 skipped, 2 warnings — 0 regressions

### Post-RPI

- Validation réelle sur DVWA/Juice Shop/WebGoat — pipeline scan complet avec findings réels
- Checklist: `Projects_reports_docs/CHECKLIST_RPI_VALIDATION_2026-05-21.md`
- Couche 2 visible dans Claude Desktop (quand le bug Anthropic sera fixé)

## Session 53 — 2026-05-21 (Phase 1 Tool Lifecycle + Phase 2 Telemetry + Bug fix str().lower())

- **Phase 1 — Tool Lifecycle** (`mcp_core/tool_registry_v2.py`): `ToolRegistry` singleton, 130 tools trackés, 92 disponibles / 38 manquants. `tool_status()` MCP tool (info + install hint). `install_tool()` MCP tool (apt/pip/gem auto-détection).
- **Phase 2 — Telemetry Pipeline** (`server_core/telemetry_pipeline.py`): `TelemetryPipeline` singleton, buffer 1000 events + per-tool aggregation + JSONL optionnel via `HEXSTRIKE_TELEMETRY_JSONL`. `finalize()` émet dans `_op_metrics` (legacy) + `_pipeline` (nouveau). 3 MCP resources: `telemetry://summary`, `telemetry://recent`, `telemetry://tools/{tool}`.
- **Bug fix `str().lower()`** : 7 corrections dans 4 fichiers. Root cause `vulnerability_correlator.py:32` (`target_software.lower()` → `str(target_software).lower()`). 6 autres préventifs. Confirmé par hexstrike.log.
- **Panel Missing Tools** (pulse_app.py) : DataTable avec Badge entre Tool Performance et Cache Status. Utilise `ToolRegistry.get_missing()`.
- **pytest.mark.slow refactor** : `test_real_integration.py` + `test_server_setup_standalone.py` marqués slow. `pytest.ini` → markers slow/flaky. `AGENTS.md` → `-m "not slow"`.
- **30 nouveaux tests** : 12 `test_tool_registry_v2.py` + 18 `test_telemetry_pipeline.py`.
- 2688 passed, 1 skipped, 2 pre-existing flaky — 0 regressions

## Session 54 — 2026-05-21 (Cache Intelligent + Profiling + scan_background)

- **`_ScanCache` refactor** : TTL scores dict, hit/miss tracking par tool, adaptation après 3+ échantillons. Hit ratio >30% → TTL ×1.2 (max 2h). Hit ratio <5% → TTL ×0.8 (min 5min). `get_ttl_scores()` et `stats()` enrichi.
- **Warmup background thread** : `_start_warmup()` appelé dans `setup_mcp_server_standalone()`. Seed version markers for first 15 installed tools via `--version`. Best-effort, daemon thread.
- **Cache Intelligence panel** (pulse_app.py) : DataTable avec Tool/Hits/Misses/Hit ratio/TTL. Entre Cache Status et Tool Performance. Rx: `CACHE_TTL`, `CACHE_TTL_SUM`. `get_cache_intelligence()` backend.
- **Benchmarks** (test_cache_benchmarks.py) : 15 benchmarks (set/get/delete/LRU/mixed/stats) + `_ScanCache` adaptive TTL + `get_ttl_scores`. Tests correctness : hit_ratio increases TTL, low hit_ratio decreases, 3-sample threshold, multi-tool.
- **Concurrent load tests** (test_cache_concurrent.py) : 9 tests — 10/20/50 threads, read/write/delete/eviction/mixed, `_ScanCache` TTL race. Barrier-synchronized, 30s timeout.
- **`import threading`** ajouté à `server_setup.py` (manquait, causait NameError au module-level).
- **`cache_ttl_scores`/`cache_ttl_summary`** unpackés dans `pulse_dashboard()` et retournés dans `_collect_dashboard_state()`.
- **`pytest-benchmark`** installé. AGENTS.md mis à jour avec commandes `--benchmark-only` et concurrent.
- **scan_background()** : nouveau `@mcp.tool(task=TaskConfig(mode="optional"), timeout=None)` dans `server_setup.py`. Async, utilise `ctx.report_progress()` entre chaque outil, retourne task_id immédiatement via le background task protocol. Description guide l'agent : "Use scan() for quick scans, scan_background() for >30s scans". 10 tests unitaires. `scan()` description mise à jour pour référencer `scan_background()`.
- **`fastmcp[tasks]`** installé (docket 0.20.1). `from fastmcp.server.tasks import TaskConfig` dans server_setup.py.
- 2700+ passed, 1 skipped, 2 pre-existing flaky — 0 regressions

## Session 55 — 2026-05-22 (Search transform refactor — RegexSearch + lean always_visible)

- **BM25SearchTransform → RegexSearchTransform** : zéro overhead (pas d'index, pas de scoring). Pattern matching prévisible. Même économie de tokens.
- **`always_visible` réduit de 10 à 4 outils** : retiré nmap, whatweb, sqlmap, get_overview, get_surface, get_findings, get_plan. Gardé seulement `scan`, `get_live_dashboard`, `scan_background`, `pulse_dashboard`. Les pentest tools et get_* individuels sont découverts via `search_tools()` — consumption purpose-driven.
- **`max_results=15` → `10`** : la search retourne 10 résultats markdown inline.
- **`list_tools()` = 6 outils** (4 always_visible + 2 meta : search_tools + call_tool) vs 12 avant.
- **Token budget** : ~6 outils fixes. Aucun pentest tool pré-chargé.
- Tests pass, import runtime OK.
- 2 files : `mcp_core/server_setup.py` (3 edits), `AGENTS.md` (this entry).
- Rapport : `Projects_reports_docs/CLAUDE_S55_BRIEF_2026-05-22.md`

## Session 55 — 2026-05-22 (Cross-session cache bug fix + 3 silent exceptions)

- **Bug fix : cross-session cache contamination** — `k.startswith(session_id)` sans `:` pouvait matcher des clés d'autres sessions. Ex : session "abc" → matche aussi "abc123:nmap:target". Fix : `f"{session_id}:"` sur 4 occurrences.
- **Bug fix : TargetStore silencieux dans scan_background()** — `except Exception: pass` → log warning.
- **Bug fix : warmup failure silencieux** — pareil, passage à `_log.warning()`.
- **Bug fix : message warning pulse_app provider** — mentionne l'impact sur always_visible (scan, get_live_dashboard, pulse_dashboard perdus).
- **Convention AGENTS.md** : bare `except Exception: pass` interdit.
- 198 tests pass, 0 regressions.

## Session 56 — 2026-05-27 (3-tab dashboard + version fix + dead state cleanup)

- **Version fix** : `config.py` `0.8.0` → `0.10.1` (décalé depuis S38, Prefab lisait la version périmée)
- **3-tab layout** : monolithique scroll remplacé par `Tabs(name="current_tab")` de Prefab. Navigation native sans click handlers.
  - **Overview** : Header → Scope → Surface → Findings → System Trends → Cache Status + Intelligence → Intelligence
  - **Workflow** : Plan IDE → Active Tools → Async Scans → Missing Tools → Rate Limit
  - **History** : History → Sessions → Errors & Failures → Tool Performance → Confirmations → Network I/O
- **Footer simplifié** : version + total runs uniquement (supprimé success rate, cache hit rate, timeouts)
- **Dead state nettoyé** : `SUCCESS_RATE`, `CACHE_HIT_RATE`, `TIMEOUT_DISP` Rx + 3 state keys supprimés
- **Panels inchangés** : même pipeline `_collect_dashboard_state()`, même données, juste réorganisation visuelle
- **Logo identifié** : `assets/hexstrike-pulse-logo.png` (1254×1254, 1.3 MB, hexagone)
- `Tab, Tabs` importés de `prefab_ui.components`
- 64/64 test_pulse_app, 131 total across 4 test files, 0 regressions

## Session 57 — 2026-05-28 (Phase 1 pipeline fixes — RPI validation via MCP)

- **Bug #1 — `task=True`** retiré de `run_security_tool` (l. 1108) + 130 typed tools (l. 1823-1829) dans `server_setup.py`. Incompatible avec opencode (pas de support task protocol). Fix : `@mcp.tool()` sans args.
- **Bug #2 — Parser ANSI** : `get_findings()` ignorait les ANSI codes nuclei. Fix : `_ANSI_RE` + `_strip_ansi()` dans `pulse_app.py`.
- **Bug #3 — Parser ports filtered** : `get_surface()` incluait les ports "filtered". Fix : ajout `parts[1] == "open"` l. 451.
- **Bug #4 — URL dans nmap** : `scan()` passait l'URL entière à nmap → "0 IP addresses". Fix : `_TOOLS_NEED_HOST` + hostname via `urlparse()` + `scan_type="-sTV"`.
- **Bug #5 — nikto -no-update** invalide → `-nocheck` dans `web_scan_direct.py`.
- **Bug #6 — Cache invalidation** des vieux résultats nmap "0 hosts up" dans `scan()`.
- **Validation RPI** : `scan(http://192.168.1.165/DVWA/)` → 2 ports (22 SSH, 80 Apache), technos (Apache/PHP), findings nuclei (1 critical, 2 medium), findings nikto (12 info).
- **opencode.jsonc timeout** 120000 → 300000.
- **Claude Desktop MCP désactivé** (conflit lock file).
- **Tests** : 64/64 pulse_app, 28/28 pipeline_integration (6 nouveaux sur ANSI/NEED_HOST/cache).
- **Rapport** : `Projects_reports_docs/CLAUDE_S57_BRIEF_2026-05-28.md`

## Session 58 — 2026-05-28 (P2 — Coverage + integration tests)

- **`telemetry_collector.py`** : 43% → **100%** (8 nouveaux tests, fichier `test_telemetry_collector.py`)
- **`tool_registry_v2.py`** : 66% → **99%** (14 nouveaux tests : auto-install, cache miss, hint branches gem/go/unknown)
- **`pulse_app.py`** : **85%** (seuil atteint, template UI exclu)
- **`mcp_entry.py`** : **96%** (déjà OK)
- **6 tests d'intégration** S57 features : `_strip_ansi()`, `_TOOLS_NEED_HOST`, cache invalidation nmap
- **TargetStore MCP resources** : déjà 28 tests, 98% coverage — aucun ajout nécessaire
- **Full suite** : 2779 passed, 1 skipped, 0 regressions
- Global coverage : **97%**

## Session 59 — 2026-05-28 (Phase 3 — ctx method gaps + state cleanup)

- **Bloc C — 3 gaps comblés** dans `test_fastmcp3_ctx_methods.py` :
  - `ctx.sample` ajouté au `make_mock_context()` (manquait, causait AttributeError si réellement invoqué)
  - `test_warning_not_called_on_successful_tool` — assert négatif que `ctx.warning` n'est pas appelé sur succès
  - `test_read_resource_mock_available` — vérifie que le mock `ctx.read_resource` est awaitable et retourne un objet avec `.contents`
  - `test_sample_mock_available` — vérifie que `ctx.sample` retourne une string mockée
- **Bloc B était mal diagnostiqué** : pas un problème d'imports circulaires. FastMCP 3.2.4 `mcp.get_prompt()` a un bug interne (kwargs passés comme `version` → `'dict' object has no attribute 'matches'`). Les fonctions prompt étaient des closures dans `register_prompts()`, pas testables directement.

- **Bloc B — Refactor + 27 tests** : Les 6 prompts sont maintenant des fonctions module-level (testables directement). `register_prompts()` les enrobe avec `mcp.prompt(name=...)`. Changement zéro risque — même signature, même registration.

  - `mcp_core/prompts.py` : 6 fonctions extraites des closures. `register_prompts()` passe de 6 `@mcp.prompt()` decorators à 6 `mcp.prompt(name=...)(fn)` appels.
  - `tests/test_prompts.py` : 27 tests. Validation :
    - Simple prompts : message count ≥ 2, first role == "user", run_security_tool présent, outils listés présents, FINAL en dernier message
    - CTF prompts : message count ≥ 3, contexte CTF, run_security_tool, stratégies/fallback, edge case workflow vide
    - Mocks : `server_core.singletons.get_ctf_manager` pour CTFWorkflowManager, `server_core.workflows.ctf.toolManager.CTFToolManager` pour CTFToolManager
    - `cloud_security_audit` : k8s_api_ip, target_image, default image

- 2811 total suite, 0 regressions.
- **Rapport débrief** : `Projects_reports_docs/CLAUDE_S59_STATE_OF_PLAY_2026-05-28.md`
- Bloc A (38 tests) + Bloc B (27 tests) + Bloc C (23 tests) + Bloc D (28 tests) = 116 tests, 0 régressions.
- Phase 3 complète : 116 tests Bloc A+B+C+D, 2811 total suite, 0 regressions.

## Session 60 — 2026-05-28 (Phase 3 — Async scan tests + pulse_app coverage 85%→90%)

- **10 nouveaux tests** dans `TestAsyncScans` (`test_pulse_app.py`) :
  - `run_async_tool` : retour immédiat, complétion background, params JSON invalide, tool inconnu, injection target, métriques
  - `get_scan_status` : not found, completed, failed with error, running with elapsed
  - Helper `wait_for_scan()` : polling loop 5s/0.05s — pas de `time.sleep(0.1)` flaky
- **Coverage pulse_app.py : 85% → 90%** (+54 lignes couvertes)
  - `run_async_tool` (1029-1096) et `get_scan_status` (1113-1147) sont maintenant à 100%
  - Les 75 lignes restantes sont du best-effort except / display / template — pas prioritaire
- `_clear_state` fixture : `pulse_app._async_scans.clear()` ajouté
- 74/74 test_pulse_app, 0 regressions.

## Session 61 — 2026-05-28 (Couche 1 — exploit_rules.py + enrichissement scan())

- **Nouveau module** `server_core/exploit_rules.py` : rules engine hybride pour enrichir les findings
  - `SPECIFIC_CVE` — 8 CVEs connus (Heartbleed, MS17-010, Log4Shell, Spring4Shell, etc.) → "certain"
  - `KEYWORD_TO_TOOL` — 32 patternes nom de vuln (SQLi→sqlmap, XSS→dalfox, RCE→metasploit, etc.)
  - `DETAIL_PATTERNS` — 10 regex sur details/path (wp-admin→wpscan, config.php→manual, .git→manual, etc.)
  - Hiérarchie : CVE > Keyword > Detail. Chaque niveau bat le suivant.
- **Modification `pulse_app.py`** : `scan()` enrichit chaque finding avec `{"exploit": {tool, confidence, estimated_time, source: "rules"}}` avant le return
- **34 tests** `test_exploit_rules.py` — coverage **100%** (chaque règle, edge cases, priorité, valeurs manquantes)
- **5 tests** `test_pulse_app.py` — vérifie que medium/full scans produisent des findings enrichis avec les bons tools
- **Correction bug** : `DETAIL_PATTERNS` ne s'exécutait pas quand `details` était vide mais `finding` avait le contenu
- **Correction bug** : `admin/` pattern trop large (matchait `/phpMyAdmin/`) — reorder patterns
- **Correction bug** : keyword matching cherchait seulement dans `finding`, pas dans `details` combiné
- 113 total nouveau tests (79 pulse_app + 34 exploit_rules), 0 regressions

## Session 62 — 2026-05-28 (RPI real target tests — Phase RPI final)

- **`tests/test_rpi.py`** : 14 tests réels contre RPI 192.168.1.165 (DVWA)
  - **Easy** (5, ~48s) : host reachable, nmap ports 22/80, whatweb Apache/PHP, scan quick + cache
  - **Medium** (4, ~6min) : pipeline 4 tools, résultats structurés, enrichissement exploit, cache persiste
  - **Hard** (5, ~7min) : gobuster paths, sqlmap banner (with session), dalfox XSS, scan full, plan web
- **Marqueur `pytest.mark.rpi`** ajouté à `pytest.ini` — skip automatique si RPI inaccessible
- **Fixtures** : `rpi_available` (nmap ping, scope="session"), `dvwa_session` (login auto CSRF), `_clear_state`
- **_populate_direct_tools()** : peuple `_DIRECT_TOOLS_CACHE` pour `scan()` sans démarrer MCP server
- **9 nouveaux tests parsing** dans `test_pulse_app.py::TestFindingsParsing` : nuclei 5 sévérités, ANSI strip, nikto paths, tri sévérité, edge cases
- 14/14 RPI tests green (7 min 19s), 88/88 pulse_app, 34/34 exploit_rules, 2869 full suite
- 0 régressions (7 pre-existing failures in test_hexstrike_server.py — HTTP server, inchangé)
- Rapport : `Projects_reports_docs/CLAUDE_REPORT_S61_FINAL_2026-05-28.md`

## Session 63 — 2026-05-28 (Phase 3 — Prompt enrichment batch 2 : wifi, smb, cloud)

- **wifi_attack_chain** enrichi : [priority] + [expected] + result_context + fallback par step. 5 steps (monitor→capture→deauth→crack→cleanup). Handshake détection guidée ("WPA handshake: {bssid}" dans output), fallback si pas de clients visibles.
- **smb_lateral_movement** enrichi : PATHS A/B — EternalBlue (unpatched) vs brute-force (patched). Decision point après STEP 3 nmap script output "VULNERABLE" vs "NOT VULNERABLE". Fallback hydra rockyou.txt + common usernames. SMB signing disabled → relay attack note.
- **cloud_security_audit** enrichi : 3 steps (prowler 30-60min, trivy 2-10min, kube_hunter 5-20min). Fallbacks : pas de creds → skip cloud audit, image not found → docker pull, K8s inaccessible → trivy config scan. Default image nginx:latest, custom image via target_image. K8s via k8s_api_ip.
- **Tests** : 27/27 test_prompts, 88/88 test_pulse_app, 34/34 test_exploit_rules — 0 regressions.
- Bloc A (PromptResult meta/tags) next — spike FastMCP 3.2.4 PromptResult compatibilité.
