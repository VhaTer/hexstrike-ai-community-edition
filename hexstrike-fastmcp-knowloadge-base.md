# HexStrike AI-PULSE — FastMCP 3.x Knowledge Base

**Last updated:** 2026-04-19 (session 8)
**Branch:** `refactor/fastmcp-modernization`
**HEAD:** `pending`
**FastMCP:** 3.2.4

---

## 🗺️ Project Overview

| | |
|---|---|
| **Project** | **HexStrike AI-PULSE** 🐻🔴 |
| **Fork** | <https://github.com/VhaTer/hexstrike-ai-community-edition> |
| **Upstream** | <https://github.com/CommonHuman-Lab/hexstrike-ai-community-edition> |
| **Upstream branch** | `beta/nullbytecobra` (tag 1.3.0) |
| **Our branch** | `refactor/fastmcp-modernization` → mergé sur `master` |
| **Venv** | `hexstrike-env/bin/python3` |
| **FastMCP** | 3.1.1 |
| **Tests** | 261/261 ✅ |
| **Philosophy** | Fork indépendant — FastMCP 3.x natif, no Flask, real-time LLM streaming |

---

## 🏗️ Architecture Phase 3 (Active)

```bash
LLM / MCP Client
    ↓ MCP protocol (HTTP/SSE)
hexstrike_server.py  →  mcp.run(transport="http", port=8888)
    ↓ run_security_tool("gobuster", '{"url": "..."}')
mcp_core/server_setup.py
    ├── ctx.read_resource("skill://web-recon/SKILL.md")   # skill context
    ├── _detect_from_cache(target)                         # auto TechProfile
    ├── _build_destructive_confirmation()                  # granular safety
    ├── _optimizer.optimize(tool, params, tech_profile)   # param optimization
    └── exec_func(tool_key, params)                        # direct execution
          ↓
    mcp_core/*_direct.py  →  server_core/command_executor.py  →  subprocess
```

---

## ✅ Ce qui est fait — Sessions 1-7

### Phase 1 — Context Migration

- 105 tools migrés à `ctx: Context`
- 83 tools avec `ctx.report_progress()`

### Phase 2 — 16 *_direct.py Modules (101+ tools)

```markdown
wifi_direct.py, recon_direct.py, net_scan_direct.py, web_scan_direct.py,
web_fuzz_direct.py, password_cracking_direct.py, smb_enum_direct.py,
exploit_framework_direct.py, web_recon_direct.py, security_direct.py,
misc_direct.py, osint_direct.py, active_directory_direct.py,
testssl_direct.py, web_probe_direct.py, vuln_intel_direct.py
```

### Phase 3 — Standalone Server (Active ✅)

- `hexstrike_server.py` → `mcp.run(transport="http", port=8888)` — zéro Flask
- `setup_mcp_server_standalone()` dans `server_setup.py`
- Resources MCP : `health://server`, `scan://{target}/latest`, `scan://{target}/{tool_name}`, `scan://cache/list`
- `@mcp.prompt()` — 5 workflow prompts : `bug_bounty_recon`, `wifi_attack_chain`, `ctf_web_challenge`, `smb_lateral_movement`, `cloud_security_audit`
- `ctx.elicit()` — elicitation sur 5 tools destructifs via `_build_destructive_confirmation()`
- `SkillsDirectoryProvider` + `BM25SearchTransform`

### IntelligentDecisionEngine ✅

- `mcp_core/technology_detector.py` — `TechProfile` + `TechnologyDetector`
- `mcp_core/parameter_optimizer.py` — profiles stealth/normal/aggressive, WAF auto-stealth
- `_detect_from_cache()` — auto TechProfile depuis whatweb/httpx cache
- `_optimizer` singleton dans `server_setup.py`

### Sécurité & Fixes (CODEX + Z.AI) ✅

- `command_executor.py` — `_executor` singleton → instance par appel (thread safety)
- `enhanced_command_executor.py` — timeout ne retourne plus `success=True`
- `enhanced_command_executor.py` — `shell=True` → argv explicit avec fallback
- `enhanced_command_executor.py` — thread leak fix (join reader threads après kill)
- `_build_destructive_confirmation()` — logique granulaire par tool :
  - `aireplay_ng` mode 9 → pas de confirmation
  - `responder` analyze=True → pas de confirmation
  - `metasploit` auxiliary/scanner|gather → pas de confirmation

### Rebrand HexStrike AI-PULSE ✅

- FastMCP server name : `hexstrike-ai-pulse`
- README public orienté utilisateur
- Logo `assets/hexstrike-pulse-logo.png`
- `master` = PULSE (`6ac0009`)

### Cleanup ✅

- `mcp_core/hexstrikecolors.py` — supprimé
- `tool_profiles.py` — stub null `HexStrikeColors` (legacy compat)

---

## 🧪 Tests — 261/261

```bash
hexstrike-env/bin/pytest tests/ -q
# 261 passed
```

| Fichier | Tests |
|---|---|
| `test_wifi_mcp_tools.py` | 37 |
| `test_osint_mcp_tools.py` | 22 |
| `test_active_directory_mcp_tools.py` | 34 |
| `test_prompts.py` | 36 |
| `test_parameter_optimizer.py` | 23 |
| `test_technology_detector.py` | 23 |
| `test_server_setup_standalone.py` | 7 (CODEX) |
| autres | 79 |

**Patterns critiques :**

- Patcher `mcp_core.module.fn` pas `server_core.command_executor.fn`
- `autouse` fixture psutil mock low-load dans `test_parameter_optimizer.py`

---

## 🧠 Key Patterns

### Tool pattern Phase 3

```python
@mcp.tool()
async def tool_name(ctx: Context, target: str) -> Dict[str, Any]:
    await ctx.info(f"🔍 Starting on {target}")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: _module_direct.exec("tool_key", data)
    )
    return result
```

### *_direct.py pattern

```python
from server_core.command_executor import execute_command

def _my_tool(data: dict) -> dict:
    err = _require(data, "target")
    if err: return err
    return execute_command(f"tool {data['target']}")

_HANDLERS = {"my_tool": _my_tool}

def module_exec(tool: str, data: dict) -> dict:
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown tool: '{tool}'"}
    return handler(data)
```

### Import critique pour les tests

```python
# ✅ Module-level — patchable
import mcp_core.wifi_direct as _wifi_direct

# ❌ From import — NOT patchable
from mcp_core.wifi_direct import wifi_exec
```

---

## 🔧 Démarrage

```bash
cd ~/Labs/hexstrike-ai-community-edition   # desktop
# ou
cd ~/hexstrike-ai-community-edition        # laptop WSL

kill $(lsof -t -i:8888) 2>/dev/null
hexstrike-env/bin/python3 hexstrike_server.py
# → HexStrike AI-PULSE sur http://127.0.0.1:8888/mcp
```

---

## 🌿 Branches

```bash
master                           ← PULSE live (= refactor HEAD)
refactor/fastmcp-modernization   ← branche de dev active
backup/phase2-complete-validated ← restore point stable
```

### Git config recommandée

```bash
git config --global pull.rebase true
git config --global rebase.autoStash true
```

### Sync desktop ↔ laptop

```bash
# Toujours au début d'une session
git pull origin refactor/fastmcp-modernization
hexstrike-env/bin/pytest tests/ -q 2>&1 | tail -3
```

---

## 🌍 Upstream

- **CommonHuman-Lab** sur `beta/nullbytecobra` (tag 1.3.0)
- Ajout `.opencode/agents/` — orchestration multi-agents OpenCode → **non pertinent pour nous**
- Nouveaux tools 1.3.0 : `hurl`, `waymore`, `assetfinder`, `shuffledns`, `massdns`, `gospider` → à cherry-pick si besoin
- **Décision** : fork indépendant, pas de rebase systématique

---

## ⚠️ Known Issues / Notes

| Issue | Solution |
|---|---|
| `_executor` singleton race condition | ✅ Fixé — instance par appel |
| timeout retournait `success=True` | ✅ Fixé |
| `shell=True` injection | ✅ Fixé — argv explicit avec fallback |
| Thread leak reader threads | ✅ Fixé — join avec timeout=2 |
| `hexstrikecolors.py` | ✅ Supprimé |
| `test_endpoints_exist.py` | Obsolète Phase 3 — exclu via `pytest.ini` |
| Desktop/laptop divergence | `git config --global rebase.autoStash true` |
| `mcp_app_direct.py` (Cline) | Invalide — ignorer/supprimer |
| `currentContext()` | N'existe pas FastMCP 3.x — utiliser `ctx: Context` |

---

## 🎯 Next Session Priorities

1. **Registre canonique unique** — gateway + standalone dérivent (5 tools standalone-only)
2. **Schema résultat normalisé** — `return_code` vs `returncode` inconsistance
3. **Push master** après chaque session significative
4. **Upstream cherry-pick** nouveaux tools 1.3.0 si pertinent

---

## 📚 FastMCP 3.x Resources

- Tools: <https://gofastmcp.com/servers/tools>
- Resources: <https://gofastmcp.com/servers/resources>
- Prompts: <https://gofastmcp.com/servers/prompts>
- Context API: <https://gofastmcp.com/servers/context>
            [Background Tasks:<https://gofastmcp.com/servers/tasks>

- Elicitation: <https://gofastmcp.com/servers/context#elicitation>
- HTTP Transport: <https://gofastmcp.com/deployment/running-server>
- SKILLS: <https://gofastmcp.com/servers/providers/skills>
- Filesystem Provider: <https://gofastmcp.com/servers/providers/filesystem>
- Local Provider : <https://gofastmcp.com/servers/providers/local>
- Lifespans: <https://gofastmcp.com/servers/lifespan>
- Progress: <https://gofastmcp.com/servers/progress>
- Storage Backends: <https://gofastmcp.com/servers/storage-backends>
- : <>