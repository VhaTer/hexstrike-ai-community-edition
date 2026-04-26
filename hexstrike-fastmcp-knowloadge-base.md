# HexStrike AI-PULSE — FastMCP 3.x Knowledge Base

**Last updated:** 2026-04-23 (session 9)
**Branch:** `refactor/fastmcp-modernization`
**HEAD:** `9cb6e88`
**FastMCP:** 3.2.4
**Tests:** 1227 passed, 6 xfailed ✅

---

## 🗺️ Project Overview

| | |
|---|---|
| **Project** | **HexStrike AI-PULSE** 🐻🔴 |
| **Fork** | <https://github.com/VhaTer/hexstrike-ai-community-edition> |
| **Upstream** | <https://github.com/CommonHuman-Lab/hexstrike-ai-community-edition> |
| **Our branch** | `refactor/fastmcp-modernization` |
| **Venv** | `hexstrike-env/bin/python3` |
| **FastMCP** | 3.2.4 |
| **Philosophy** | Fork indépendant — FastMCP 3.x natif, no Flask, real-time LLM streaming |

### Machines

| Machine | Path | OS |
|---|---|---|
| Laptop | `~/hexstrike-ai-community-edition` | Kali Linux WSL |
| Desktop (PureHate) | `~/Labs/hexstrike-ai-community-edition` | Kali Linux natif |

---

## 🏗️ Architecture Phase 3 (Active)

```
LLM / MCP Client
    ↓ MCP protocol (HTTP/SSE)
hexstrike_server.py  →  mcp.run(transport="http", port=8888, show_banner=False)
    ↓
mcp_core/server_setup.py — setup_mcp_server_standalone()
    ├── SkillsDirectoryProvider(skills/, supporting_files="resources", reload=True)
    ├── BM25SearchTransform — tool discovery
    ├── run_security_tool()  — generic entrypoint
    ├── get_tool_skill()     — skill bundle helper
    ├── typed MCP tool wrappers (auto-générés depuis tool_registry.py)
    ├── Resources: health://server, scan://{target}/latest|{tool}, scan://cache/list
    └── Prompts: 5 workflow prompts (bug_bounty_recon, wifi_attack_chain, ...)
          ↓
    mcp_core/*_direct.py  →  server_core/command_executor.py  →  subprocess
```

---

## ✅ Features MCP Context — Complètes

| Feature | Implémentation | Notes |
|---|---|---|
| `ctx.info()` / `ctx.error()` | ✅ run_security_tool | Streaming vers LLM |
| `ctx.report_progress()` | ✅ run_security_tool | Phases 0/25/50/75/100%, tick 12s |
| `ctx.elicit()` | ✅ elicitation.py | 5 tools destructifs, `response_type=bool` |
| `ctx.read_resource()` | ✅ server_setup.py | SKILL.md injecté avant exécution |
| `ctx.set_state()` / `ctx.get_state()` | ✅ server_setup.py | TechProfile persisté par session/target |
| LLM Sampling | — hors scope | Nécessite client supportant sampling |

### ctx.set_state() — comportement

```python
# Lecture : session state d'abord (plus rapide), puis scan cache
cached_dict = await ctx.get_state(f"tech:{target}")
# → reconstruit TechProfile depuis dict JSON

# Écriture : après détection depuis scan cache
await ctx.set_state(f"tech:{target}", {
    "web_servers": [...], "frameworks": [...], "cms": [...], ...
})
# Graceful degradation : except Exception: pass si session indisponible
```

---

## ✅ Skills System

- **14 skill bundles** dans `skills/` : `nmap-recon`, `web-recon`, `web-vuln`, `wifi-pentest`, `active-directory`, `exploitation`, `smb-enum`, `subdomain-enum`, `osint-recon`, `password-cracking`, `binary-analysis`, `cloud-audit`, `hexstrike-workflows`, `testssl`
- **91 tool→skill mappings** dans `_TOOL_SKILL_MAP`
- `SkillsDirectoryProvider(roots=skills/, supporting_files="resources", reload=True)` — exposé via `skill://...`
- `get_tool_skill(tool_name)` — MCP tool retournant le bundle complet
- `_read_skill_document()` / `_read_skill_bundle()` — extraction via `ResourceResult.contents[0].content`
- **`hexstrike-workflows`** — index des 5 prompts MCP, orchestre les autres skills

---

## ✅ 16 *_direct.py Modules

```
wifi_direct.py          recon_direct.py         net_scan_direct.py
web_scan_direct.py      web_fuzz_direct.py      password_cracking_direct.py
smb_enum_direct.py      exploit_framework_direct.py  web_recon_direct.py
security_direct.py      misc_direct.py          osint_direct.py
active_directory_direct.py  testssl_direct.py   web_probe_direct.py
vuln_intel_direct.py
```

**106 direct routes, 105 typed MCP tool wrappers** (auto-générés depuis `tool_registry.py`)

### Sécurité : web_scan_direct.py

`_sanitize()` appliqué sur `custom_payload` dans `_dalfox` — strip des caractères dangereux (`'`, `` ` ``, `;`, `&&`, `||`, `$(`, `${`)

---

## 🧪 Tests

```bash
hexstrike-env/bin/pytest tests/ -q
# 1227 passed, 6 xfailed
```

**Pattern de mock critique :**

```python
# ✅ Patcher là où le nom est utilisé
with patch("mcp_core.osint_direct.execute_command") as mock: ...

# ❌ Patcher la source — ne fonctionne pas
with patch("server_core.command_executor.execute_command") as mock: ...
```

**Tests notables :**
- `test_server_setup_standalone.py` — elicitation + destructive tool confirmation
- `test_parameter_optimizer.py` — psutil mock `autouse` fixture requis

---

## 🧠 Patterns Clés

### run_security_tool — flux complet

```python
@mcp.tool()
async def run_security_tool(ctx: Context, tool_name: str, parameters: str):
    # 1. Skill guidance via ctx.read_resource()
    # 2. Elicitation si tool destructif
    # 3. TechProfile : session state → scan cache → None
    # 4. ParameterOptimizer (WAF stealth, resource tuning)
    # 5. run_in_executor + asyncio.wait (progress phases)
    # 6. Cache result + ctx.set_state(tech_profile)
```

### Import pattern — critique pour les tests

```python
# ✅ Module-level — patchable
import mcp_core.wifi_direct as _wifi_direct

# ❌ From import — NOT patchable
from mcp_core.wifi_direct import wifi_exec
```

### ctx.elicit() — pattern validé

```python
result = await ctx.elicit(message, response_type=bool)
# response_type obligatoire en FastMCP 3.2.4 (déprécié sans)
if hasattr(result, "action"):
    if result.action == "accept" and result.data is True: ...
```

### CurrentContext()

Existe dans `fastmcp.server.dependencies` comme **DI helper**. Le framework l'injecte automatiquement via `transform_context_annotations()` pour tout `ctx: Context` sans default. `ctx: Context` seul est suffisant et équivalent.

---

## 🔧 Démarrage

```bash
cd ~/hexstrike-ai-community-edition   # laptop
# ou ~/Labs/hexstrike-ai-community-edition  # desktop

kill $(lsof -t -i:8888) 2>/dev/null
hexstrike-env/bin/python3 hexstrike_server.py
# → HexStrike AI-PULSE banner + http://127.0.0.1:8888/mcp
```

### Startup — ce qui s'affiche

```
[HexStrike ANSI banner via ModernVisualEngine.create_banner()]
[INFO] 🚀 Starting HexStrike Pulse Standalone Server
[INFO] 🤖 Skills initialized
[INFO] 🚀 Phase 3: 106 direct routes, 105 typed MCP tools
[INFO] 📦 Resources MCP registered
[INFO] 🎯 Workflow prompts registered
```

FastMCP banner supprimée via `show_banner=False` dans `mcp.run()`.

---

## 🌿 Git

```bash
# Branches
refactor/fastmcp-modernization   ← dev active
backup/phase2-complete-validated ← restore point

# Début de session — toujours
git pull origin refactor/fastmcp-modernization
hexstrike-env/bin/pytest tests/ -q 2>&1 | tail -3

# Config recommandée
git config --global pull.rebase true
git config --global rebase.autoStash true
```

---

## 🌍 Upstream & Outils externes

- **CommonHuman-Lab** — fork indépendant, pas de rebase systématique
- `*_direct.py` sont entièrement notre travail — jamais dans upstream
- **Codex** — bon pour skills/refactoring, vérifier la syntax FastMCP sur source
- **Perplexity/GPT** — délégué à pytest-cov coverage
- **Z.AI** — review externe utile, même sans contexte complet
- Docs AI générées (`*_REPORT.md`, `*_ANALYSIS.md`) → gitignorées, lire avant d'ignorer

---

## ⚠️ Known Issues / Notes

| Issue | Solution |
|---|---|
| Thread safety `_executor` | ✅ instance par appel |
| Timeout retournait `success=True` | ✅ fixé |
| `shell=True` injection | ✅ argv explicit |
| Thread leak reader threads | ✅ join avec timeout=2 |
| `test_endpoints_exist.py` | Obsolète — exclu via `pytest.ini` |
| Desktop/laptop divergence | `git config --global rebase.autoStash true` |
| `ctx.set_state()` indisponible | Graceful degradation — `except Exception: pass` |
| Dashboard `/web-dashboard` | Routes Flask legacy — long-term item |

---

## 🧠 IntelligentDecisionEngine — Integration Policy

**Statut :** Advisory uniquement — jamais authoritative

| Question | Réponse |
|---|---|
| Advisory ou authoritative ? | **Advisory** — le LLM décide toujours de l'exécution |
| Planifie tools, params, ou les deux ? | **Tools uniquement** — params restent dans ParameterOptimizer |
| Où vit le human override ? | `ctx.elicit()` sur chaque step destructif dans `run_security_tool` |
| Confirmation boundaries sur les chains ? | Chaque step de l'AttackChain passe par `run_security_tool` — les gates existantes s'appliquent |
| `TargetProfile` / `AttackChain` comme runtime contracts ? | **Non pour l'instant** — ce sont des suggestions JSON retournées au LLM |

**Pattern d'usage :**
```
LLM → plan_attack(target, objective)   # IDE analyse + suggère
    → AttackChain JSON retourné au LLM
    → LLM décide quels steps exécuter
    → run_security_tool() pour chaque step
    → gates elicitation/skill/progress s'appliquent normalement
```

**Ce qu'on n'implémentera PAS sans validation supplémentaire :**
- Exécution automatique d'une AttackChain sans confirmation LLM step-by-step
- Override du ParameterOptimizer par l'IDE
- AttackChain comme contrat d'exécution mandatory

---

## 📚 FastMCP 3.x Resources

- Context API + State: <https://gofastmcp.com/servers/context>
- Skills Provider: <https://gofastmcp.com/servers/providers/skills>
- Tools: <https://gofastmcp.com/servers/tools>
- Resources: <https://gofastmcp.com/servers/resources>
- Prompts: <https://gofastmcp.com/servers/prompts>
- Elicitation: <https://gofastmcp.com/servers/context#elicitation>
- HTTP Transport: <https://gofastmcp.com/deployment/running-server>
- Settings: <https://gofastmcp.com/more/settings>
- Storage Backends: <https://gofastmcp.com/servers/storage-backends>
