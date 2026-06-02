# Claude — Brief Session 53 : Tool Lifecycle + Telemetry Unifiée

Date : 2026-05-21
Auteur : opencode
Contexte : relecture du log hexstrike.log — merci de rapporter le bug que tu as spoté.

---

## Terminologie

**Pulse** = le projet HexStrike AI-PULSE dans son ensemble (MCP server, 162 tools, dashboard Prefab, pipeline data, tool registry, telemetry).  
**pulse_app.py** = UN fichier parmi d'autres (le entry point UI Prefab). Ne pas confondre.

---

## Ce qui a été fait (Session 52-53)

### Phase 1 — Tool Lifecycle (`mcp_core/tool_registry_v2.py`)

**Nouveau fichier.** Aucun autre fichier modifié (sauf server_setup.py pour les imports + 2 MCP tools).

- `ToolRegistry` singleton : wrap `TOOL_ROUTES` (130 tools) + `INSTALL_HINTS` (50 outils courants)
- `get_tool_status(name)` → installed/not_found/unknown via `shutil.which()`
- `get_available()` / `get_missing()` → partition des outils
- `install(name)` → apt/pip/gem auto-détecté, timeout 120s
- 4 alias exclus (`nmap_advanced`, `bettercap_wifi`, `nxc`, `wifite2`)

**Décision architecturale importante** : `shutil.which(binary)` est INFIABLE pour les frameworks.
metasploit → binary "metasploit" n'existe pas (msfconsole). bloodhound idem.
J'avais tenté une graceful degradation (skip typed wrapper + early return dans run_security_tool)
mais ca cassait `test_metasploit_auxiliary_scanner_skips_confirmation`.

→ **Revert** : TypedRegistry est purement INFORMATIF, pas bloquant.
Tous les outils restent accessibles. `tool_status()` informe l'agent qui décide.

### Phase 2 — Telemetry Unifiée (`server_core/telemetry_pipeline.py`)

**Nouveau fichier.** Aucun changement dans `operational_metrics.py` (backward compat).

- `TelemetryPipeline` singleton : 3 backends (buffer 1000 events + per-tool aggregation + JSONL optionnel)
- Thread-safe via RLock
- `finalize()` dans `server_setup.py` écrit maintenant dans DEUX stores :

```python
_op_metrics.record(_telemetry)   # ← existant, dashboard, ressources, legacy
_pipeline.emit(_telemetry)       # ← nouveau, data pipeline
```

Aucun consumer existant touché. Migration future panel par panel.

### MCP Tools ajoutés

```
tool_status(tool_name="") → statut outil ou summary global
install_tool(tool_name)   → apt/pip/gem, timeout 180s
```

### MCP Resources ajoutées

```
telemetry://summary        → runs, errors, cache, confirmations
telemetry://recent         → 100 derniers events
telemetry://tools/{tool}   → stats par outil
```

### Refactor Tests

```
tests/test_real_integration.py       → pytest.mark.slow
tests/test_server_setup_standalone.py → pytest.mark.slow
pytest.ini  → markers = slow, flaky
AGENTS.md   → -m "not slow" (remplace --ignore)
```

### Tests

- 12 tests `test_tool_registry_v2.py` — nouveau
- 18 tests `test_telemetry_pipeline.py` — nouveau
- **2659 passed, 1 skipped, 2 warnings — 0 regressions**

---

## Ce qui N'A PAS changé

- `pulse_app.py` — PAS TOUCHÉ (zéro changement dans le dashboard Prefab)
- `operational_metrics.py` — PAS TOUCHÉ (backward compat intégrale)
- `hexstrike_middleware.py` — PAS TOUCHÉ (le pipeline est alimenté par finalize(), pas par le middleware)
- `server_setup.py` — import + finalize() + 2 tools + telemetry resources. Rien d'autre.
- Tous les autres fichiers — AUCUN CHANGEMENT.

---

## À discuter

- Le bug que tu as trouvé dans hexstrike.log ?
- Dashboard Panel "Missing Tools" (ajout Prefab) — pas fait, prévu si pertinent
- Prochaine direction : soit le panel, soit autre chose
