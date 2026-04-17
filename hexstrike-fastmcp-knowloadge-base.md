# HexStrike CE — FastMCP 3.x Knowledge Base

**Last updated:** 2026-04-17 (session 7)
**Branch:** `refactor/fastmcp-modernization`
**HEAD:** `249aac9`
**FastMCP:** 3.1.1

---

## 🗺️ Project Overview

| | |
|---|---|
| **Fork** | <https://github.com/VhaTer/hexstrike-ai-community-edition> |
| **Upstream** | <https://github.com/CommonHuman-Lab/hexstrike-ai-community-edition> |
| **Community Edition** | CommonHuman-Lab (active dev on `beta/rootkitfox`) |
| **Our fork** | VhaTer — FastMCP 3.x refactor |
| **Venv** | `hexstrike-env/bin/python3` |
| **FastMCP** | 3.1.1 |
| **Tests** | 129/129 ✅ (37 wifi + 22 osint + 34 AD + 36 prompts) |

---

## ✅ What We Built — Phase 1, 2 & 3

### Phase 1 — Context Migration

- **105 tools** migrated to `ctx: Context`
- **83 tools** with `ctx.report_progress()`
- `ctx.info()` / `ctx.error()` streams to LLM in real time

### Phase 2 — 13 *_direct.py Modules (101 tools routed)

```
mcp_core/
├── wifi_direct.py              # 12 tools — wifi_pentest
├── recon_direct.py             # 7 tools  — recon, dns_enum, net_lookup
├── net_scan_direct.py          # 5 tools  — nmap, masscan, rustscan, arp-scan
├── web_scan_direct.py          # 7 tools  — nikto, sqlmap, wpscan, dalfox...
├── web_fuzz_direct.py          # 7 tools  — gobuster, ffuf, feroxbuster...
├── password_cracking_direct.py # 8 tools  — hydra, hashcat, john, hashid...
├── smb_enum_direct.py          # 6 tools  — enum4linux, netexec, smbmap...
├── exploit_framework_direct.py # 5 tools  — metasploit, msfvenom, pwntools...
├── web_recon_direct.py         # 9 tools  — katana, gau, httpx, arjun...
├── security_direct.py          # 11 tools — cloud, container, k8s, iac
├── misc_direct.py              # 28 tools — binary, forensics, db, api_scan...
├── osint_direct.py             # 4 tools  — sherlock, spiderfoot, sublist3r, parsero
└── active_directory_direct.py  # 8 handlers — impacket, ldapdomaindump, adidnsdump,
                                #              certipy_ad, mitm6, pywerview, bloodhound
```

**101 tools dans DIRECT_ROUTES** (gateway.py) + **DIRECT_TOOLS** (server_setup.py).

### Phase 3 — Standalone Server + Elicitation + Resources (ACTIF ✅)

- `hexstrike_server.py` → `mcp.run(transport="http", port=8888)` — zéro Flask
- `setup_mcp_server_standalone()` dans `server_setup.py`
- FastMCP 3.1.1, HTTP/SSE transport natif
- `ctx.elicit()` — `mcp_core/elicitation.py` → `confirm_destructive_action()` sur 5 outils
- Resources MCP — 4 resources : `health://server`, `scan://{target}/latest`, `scan://{target}/{tool_name}`, `scan://cache/list`
- Cache in-memory `_scan_cache` dans `server_setup.py` — alimenté par `run_security_tool`
- `theHarvester` fix — clé `"theharvester"` dans `DIRECT_TOOLS` ✅
- `@mcp.prompt()` — 5 workflow prompts natifs dans `mcp_core/prompts.py`
- Skills MCP — `SkillsDirectoryProvider(..., supporting_files="resources", reload=True)`
- `get_tool_skill(tool_name)` — récupère `SKILL.md` + supporting docs par mapping outil → skill
- Typed MCP tools — wrappers auto-générés depuis `tool_registry.py` pour les outils directs ayant un schéma compact

---

## 🏗️ Architecture

### Phase 3 (actuelle)

```
Claude / MCP client
    ↓ MCP protocol
hexstrike_server.py → mcp.run(transport="http", port=8888)
    ↓ run_security_tool("nmap", '{"target": "..."}')
mcp_core/*_direct.py → execute_command()
```

### Legacy (hexstrike_mcp.py — profile-based, toujours fonctionnel)

```
hexstrike_mcp.py → HexStrikeClient → DIRECT_ROUTES → *_direct.py
                                    → Flask fallback (tools non migrés)
```

---

## 🔄 Phase 3 — TODO restant

### 1. Encore sur Flask legacy — skip Phase 3

```
ops/ (8), web_framework/ (2), ai_assist/, bugbounty_workflow/,
error_handling/, ai_payload/, web_scan/burpsuite
```

---

## 🔬 FastMCP 3.x Features — Status

| Feature | Status | Notes |
|---|---|---|
| `ctx.info()` / `ctx.error()` | ✅ 105 tools | Streams to LLM |
| `ctx.report_progress()` | ✅ 83 tools | Non-blocking via asyncio.wait |
| `BM25SearchTransform` | ✅ server_setup.py | Tool discovery |
| `SkillsDirectoryProvider` | ✅ server_setup.py | Skills exposed as resources with reload enabled |
| `run_in_executor` | ✅ All tools | Non-blocking I/O |
| Phase 3 standalone server | ✅ ACTIF | mcp.run(transport="http") |
| `ctx.elicit()` | ✅ ACTIF | `mcp_core/elicitation.py` — 5 outils destructifs |
| Resources MCP | ✅ ACTIF | 4 resources dans server_setup.py |
| `@mcp.prompt()` | ✅ ACTIF | 5 prompts dans `mcp_core/prompts.py` |
| `get_tool_skill()` | ✅ ACTIF | Runtime access to skill bundles |
| Typed tool wrappers | ✅ ACTIF | Auto-registered from compact tool registry metadata |
| LLM Sampling | ❌ Not yet | Claude Desktop pas encore |

---

## 🧠 Key Patterns

### Tool pattern (complet)

```python
from fastmcp import Context
import asyncio
import mcp_core.wifi_direct as _wifi_direct  # module-level — patchable en tests

@mcp.tool()
async def tool_name(ctx: Context, target: str, ...) -> Dict[str, Any]:
    data = {"target": target, ...}
    await ctx.info(f"🔍 Starting on {target}")
    await ctx.report_progress(0, 100)

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None, lambda: _wifi_direct.wifi_exec("tool_key", data)
    )

    phases = [(25, "Phase 1..."), (50, "Phase 2..."), (75, "Phase 3...")]
    tick = 15
    for progress, message in phases:
        done, _ = await asyncio.wait([future], timeout=tick)
        if done: break
        await ctx.report_progress(progress, 100)
        await ctx.info(message)

    result = await future
    await ctx.report_progress(100, 100)
    if result.get("success"):
        await ctx.info("✅ Done")
    else:
        await ctx.error(f"❌ Failed: {result.get('error')}")
    return result
```

**IMPORTANT :**

- `ctx: Context` injecté automatiquement par type hint reste valide
- `CurrentContext()` **existe** dans FastMCP 3.1.1 mais dans `fastmcp.server.dependencies` — c'est un **helper de DI** (dependency injection) utilisé comme valeur par défaut : `ctx: Context = CurrentContext()`. Le framework l'injecte **automatiquement** via `transform_context_annotations()` sur tout paramètre `ctx: Context` sans default. Donc écrire `ctx: Context` suffit — `CurrentContext()` explicite est redondant mais valide. Ne pas confondre avec un appel runtime `CurrentContext()` hors d'une tool function (lève RuntimeError).

### Skills pattern — actuel

Les skills ne sont plus juste des playbooks monolithiques. Le repo les expose comme bundles filesystem:

```text
skills/<skill-name>/
├── SKILL.md
└── REFERENCE.md
```

Notes:

- `SKILL.md` = quand utiliser la skill + workflow opératoire
- `REFERENCE.md` = syntaxe typed MCP tool calls (`nmap(...)`, `sqlmap(...)`, etc.) + fallback `run_security_tool(...)`
- `SkillsDirectoryProvider(..., supporting_files="resources")` expose ces fichiers via `skill://...`
- `get_tool_skill(tool_name)` permet de récupérer le bundle local associé à un outil

### Skill runtime pattern

```python
@mcp.tool
async def get_tool_skill(ctx: Context, tool_name: str) -> Dict[str, Any]:
    skill_name = _TOOL_SKILL_MAP.get(tool_name.lower())
    documents = await _read_skill_bundle(ctx, skill_name)
    return {
        "success": True,
        "tool_name": tool_name,
        "skill_name": skill_name,
        "documents": documents,
        "available_files": sorted(documents),
    }
```

### *_direct.py pattern

```python
from server_core.command_executor import execute_command

def _require(data: dict, *keys) -> dict:
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"'{key}' is required"}
    return {}

def _my_tool(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    return execute_command(f"tool {data['target']}")

_HANDLERS = {"my_tool": _my_tool}

def example_exec(tool: str, data: dict) -> dict:
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown tool: '{tool}'"}
    return handler(data)
```

### Import pattern — critique pour les tests

```python
# ✅ Module-level — patchable
import mcp_core.wifi_direct as _wifi_direct

# ❌ From import — NOT patchable
from mcp_core.wifi_direct import wifi_exec
```

### Patch pattern dans les tests

```python
# ✅ Patcher là où le nom est utilisé (pas la source)
with patch("mcp_core.osint_direct.execute_command") as mock: ...
with patch("mcp_core.active_directory_direct.ad_exec") as mock: ...

# ❌ Patcher la source ne fonctionne pas après import
with patch("server_core.command_executor.execute_command") as mock: ...
```

---

## 🧪 Testing — 93 tests, 24 classes

```bash
# Tous les tests (pytest.ini exclut test_endpoints_exist.py automatiquement)
hexstrike-env/bin/pytest tests/ -q

# Par module
hexstrike-env/bin/pytest tests/test_wifi_mcp_tools.py -v           # 37 tests
hexstrike-env/bin/pytest tests/test_osint_mcp_tools.py -v          # 22 tests
hexstrike-env/bin/pytest tests/test_active_directory_mcp_tools.py -v # 34 tests
```

**test_endpoints_exist.py** — teste routes Flask, obsolète Phase 3.
Exclu via `pytest.ini` (`addopts = --ignore=tests/test_endpoints_exist.py`).

---

## 🌿 Branch Structure

```
refactor/fastmcp-modernization   ← active development
backup/phase2-complete-validated ← stable restore point — synced HEAD
feat/wifi-pentest-clean          ← gardée, merged upstream
```

### Git commands utiles

```bash
# Avant tout rebase — analyser le diff
git fetch upstream
git log --oneline upstream/beta/rootkitfox ^HEAD | head -30
git diff --name-only upstream/beta/1.0.13..upstream/beta/rootkitfox
git diff HEAD..upstream/beta/rootkitfox -- mcp_core/ server_core/ \
    tool_registry.py mcp_core/tool_profiles.py > /tmp/diff_critique.md

# Stash session notes avant rebase
git stash push -u -m "session-notes-pre-rebase"

# Rebase pattern rootkitfox
git tag pre-rebase-$(date +%Y%m%d) HEAD
git rebase upstream/beta/rootkitfox
# Conflits sur nos *_direct.py → toujours --ours
git checkout --ours mcp_core/exploit_framework_direct.py
git checkout --ours mcp_core/misc_direct.py
git checkout --ours mcp_core/mcp_entry.py   # garder log_level="WARNING"
git add <fichiers> && GIT_EDITOR=true git rebase --continue
git push --force-with-lease origin refactor/fastmcp-modernization

# Update backup
git branch -f backup/phase2-complete-validated HEAD
git push origin backup/phase2-complete-validated --force-with-lease
```

---

## ⚠️ Known Issues / Notes

- The standalone server still exposes one generic execution tool, `run_security_tool(tool_name, parameters)`, rather than fully typed MCP tools per security utility. This keeps Phase 3 stable but means skill references remain the main place for per-tool invocation guidance.
- Skill examples should prefer typed MCP tools when available, with `run_security_tool(...)` used as the fallback for advanced or not-yet-modeled cases.
- Anthropic-style "agent skill" semantics and FastMCP resource-provider semantics are related but not identical:
  - Anthropic / Claude Code: filesystem skill bundle loaded progressively by the agent
  - FastMCP: skill bundle exposed as MCP resources/templates, then consumed by clients or helper tools

| Issue | Solution |
|---|---|
| `testing.py` WinDefender | Stub GPT-5 en place ✅ |
| `GIT_EDITOR=true` | Évite vim sur rebase --continue |
| ANSI colors `HexStrikeColors` | Casse JSON parsing — retirer Phase 3 cleanup |
| `mcp_entry.py` log_level | rootkitfox supprime `log_level="WARNING"` → toujours `--ours` |
| `currentContext()` | N'existe pas FastMCP 3.x — utiliser `ctx: Context` type hint |
| Patch tests | Patcher `mcp_core.module.fn` pas `server_core.command_executor.fn` |
| `test_endpoints_exist.py` | Obsolète Phase 3 — exclu via `pytest.ini` |
| 3 tools FastMCP internes | `trace`, `fail`, `sleep` — injectés FastMCP 3.1.1, inoffensifs |
| Stash double rebase | `git stash pop` x2 — conflit `.gitignore` normal, résoudre `--ours` |
| `theHarvester` casse | ✅ Fixé — clé `"theharvester"` dans `DIRECT_TOOLS` |

---

## 🌍 Upstream / Contributions externes

### GPT-5 contributions (session 3, 2026-03-28)

- `hexstrike_server.py` Phase 3 standalone (bug `asyncio` corrigé par nous)
- `server_setup.py` `setup_mcp_server_standalone()` (corrigé)
- `testing.py` stub `FileUploadTestingFramework`
- `AGENTS.md` doc agent (corrigé : `CurrentContext()` retiré, chiffres tests)
- `STRATEGIC_REFLECTION.md` — bonne doc vision, garder

### Différenciateurs vs CommonHuman-Lab

1. `ctx: Context` → LLM voit progress en temps réel
2. 13 `*_direct.py` → No Flask (~150 HTTP calls éliminés)
3. `run_in_executor` → Non-blocking async
4. `gateway DIRECT_ROUTES` → 101 tools routés directement
5. 93 tests (37 wifi + 22 osint + 34 AD)
6. `BM25SearchTransform` → Smart tool discovery
7. Phase 3 standalone server actif — zéro Flask

---

## 🚀 Production Setup

### Démarrer le serveur Phase 3

```bash
cd ~/hexstrike-ai-community-edition
kill $(lsof -t -i:8888) 2>/dev/null
hexstrike-env/bin/python3 hexstrike_server.py
# → FastMCP 3.1.1 sur http://127.0.0.1:8888/mcp
```

### Antigravity IDE (hexstrike_mcp.py — profile-based)

```json
{
  "mcpServers": {
    "hexstrike": {
      "command": "wsl.exe",
      "args": ["-d", "kali-linux", "--",
        "/home/vhater/hexstrike-ai-community-edition/hexstrike-env/bin/python3",
        "/home/vhater/hexstrike-ai-community-edition/hexstrike_mcp.py",
        "--server", "http://127.0.0.1:8888",
        "--profile", "full"
      ],
      "timeout": 300
    }
  }
}
```

### Validate

```bash
hexstrike-env/bin/pytest tests/ -q 2>&1 | tail -3
hexstrike-env/bin/python3 -c "
import mcp_core.osint_direct
import mcp_core.active_directory_direct
print('✅ All direct modules OK')
"
```

---

## 📚 FastMCP 3.x Resources

- Context API: <https://gofastmcp.com/servers/context>
- Elicitation: <https://gofastmcp.com/servers/context#elicitation>
- Prompts: <https://gofastmcp.com/servers/prompts>
- Resources: <https://gofastmcp.com/servers/resources>
- HTTP Transport: <https://gofastmcp.com/deployment/running-server>

---

## 🔬 Elicitation — Détails implémentation

### `mcp_core/elicitation.py`

```python
from mcp_core.elicitation import confirm_destructive_action

confirmed = await confirm_destructive_action(
    ctx,
    action="Deauth attack on AA:BB:CC",
    detail="interface: wlan0mon",
    warning="Will disconnect all clients"
)
if not confirmed:
    return {"success": False, "error": "Cancelled by user"}
```

**Outils protégés :** `aireplay_ng`, `responder`, `metasploit`, `mdk4`, `mitm6`
**Fallback :** `NotImplementedError` → action bloquée + message explicite (Cursor, VS Code)
**Supporté :** Claude Desktop ✅, Antigravity IDE ✅

### Resources MCP — Détails

| Resource | URI | Description |
|---|---|---|
| `server_health` | `health://server` | Uptime, tool count, cache stats |
| `scan_latest` | `scan://{target}/latest` | Dernier scan pour un target |
| `scan_result` | `scan://{target}/{tool_name}` | Résultat spécifique tool+target |
| `scan_cache_list` | `scan://cache/list` | Liste de tous les scans cachés |

Cache alimenté automatiquement par `run_security_tool()` à chaque succès.

---

## 🎯 Next Session Priorities

1. **`ctx.read_resource()`** — implémenter dans les tools (lire résultats de scans précédents)
2. **Rebase upstream/beta/rootkitfox** — vérifier nouveaux tools CommonHuman-Lab
3. **PR** vers CommonHuman-Lab
4. **Cleanup** — retirer `HexStrikeColors` ANSI (casse JSON parsing)
