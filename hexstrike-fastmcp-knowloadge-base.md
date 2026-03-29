# HexStrike CE — FastMCP 3.x Knowledge Base

**Last updated:** 2026-03-28 (session 4)
**Branch:** `refactor/fastmcp-modernization`
**HEAD:** voir `git log --oneline -1`
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
| **Tests** | 93/93 ✅ (37 wifi + 22 osint + 34 active_directory) |

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

### Phase 3 — Standalone Server (ACTIF ✅)

- `hexstrike_server.py` → `mcp.run(transport="http", port=8888)` — zéro Flask
- `setup_mcp_server_standalone()` dans `server_setup.py`
- FastMCP 3.1.1, HTTP/SSE transport natif

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

### 1. User Elicitation — actions destructives (priorité haute)

```python
result = await ctx.elicit(
    "Confirm deauth attack on AA:BB:CC:DD:EE:FF?",
    response_type=bool
)
if result.action == "accept" and result.data:
    # run aireplay_ng
```

**Cibles :** `aireplay_ng`, `responder`, `metasploit`, `mdk4`, `mitm6`
**Note :** Supporté Claude Desktop ✅

### 2. Migrer les agents en `@mcp.prompt()`

12 agents dans `server_core/intelligence/` et `server_core/workflows/` :
`IntelligentDecisionEngine`, `BugBountyWorkflowManager`, `CTFWorkflowManager`,
`CVEIntelligenceManager`, `AIExploitGenerator`, `VulnerabilityCorrelator`,
`TechnologyDetector`, `RateLimitDetector`, `FailureRecoverySystem`,
`PerformanceMonitor`, `ParameterOptimizer`, `GracefulDegradation`

### 3. Resources MCP

```python
@mcp.resource("scan://{target}/results")
async def scan_result(target: str) -> str:
    return get_cached_scan(target)

@mcp.resource("health://server")
async def server_health() -> str:
    return get_health_json()
```

### 4. theHarvester fix

`recon_direct.py` — synchroniser la majuscule upstream : `theharvester` → `theHarvester`

### 5. Encore sur Flask legacy — skip Phase 3

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
| `SkillsDirectoryProvider` | ✅ server_setup.py | Skills as resources |
| `run_in_executor` | ✅ All tools | Non-blocking I/O |
| Phase 3 standalone server | ✅ ACTIF | mcp.run(transport="http") |
| `@mcp.prompt()` | 🔄 TODO | Workflows agents |
| Resources MCP | 🔄 TODO | Scan results, health |
| User Elicitation | 🔄 TODO | Actions destructives |
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

**IMPORTANT :** `ctx: Context` injecté automatiquement par type hint.
Ne PAS utiliser `ctx: Context = CurrentContext()`.

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
| `theHarvester` casse | À sync dans `recon_direct.py` — `theharvester` → `theHarvester` |

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

## 🎯 Next Session Priorities

1. **User Elicitation** — `aireplay_ng`, `metasploit`, `responder`, `mdk4`, `mitm6`
2. **`theHarvester` fix** — `recon_direct.py` sync majuscule upstream
3. **Resources MCP** — `health://server` + `scan://{target}/results`
4. **`@mcp.prompt()`** — porter `BugBountyWorkflowManager` en prompt natif
5. **PR** vers CommonHuman-Lab
