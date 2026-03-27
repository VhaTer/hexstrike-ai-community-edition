# HexStrike CE — FastMCP 3.x Knowledge Base

**Last updated:** 2026-03-27  
**Branch:** `refactor/fastmcp-modernization`  
**HEAD:** `395375a`  
**Tag:** `phase2-validated`

---

## 🗺️ Project Overview

| | |
|---|---|
| **Fork** | <https://github.com/VhaTer/hexstrike-ai-community-edition> |
| **Upstream** | <https://github.com/CommonHuman-Lab/hexstrike-ai-community-edition> |
| **Original** | 0x4m4 |
| **Community Edition** | CommonHuman-Lab (active dev on `beta/1.0.13`) |
| **Our fork** | VhaTer — FastMCP 3.x refactor |
| **Venv** | `hexstrike-env/bin/python3` |
| **Tests** | `hexstrike-env/bin/pytest tests/test_wifi_mcp_tools.py` → 37/37 ✅ |

---

## ✅ What We Built — Phase 1 & 2

### Phase 1 — Context Migration

- **40+** tools explicitly using `ctx: Context` parameter (FastMCP 3.x dependency injection)
- **70+** tools calling `ctx.info()` / `ctx.error()` for structured logging to LLM
- **15-20** tools with `ctx.report_progress()` on long-running operations (web_fuzz, scanning)
- **Pattern:** `asyncio.wait([future], timeout=tick)` — non-blocking progress
- `ctx.info()` / `ctx.error()` streams to LLM in real time (all security tools)

### Phase 1 — wifi_direct.py

- First `*_direct.py` module — bypasses Flask entirely
- 12 WiFi tool handlers — pure Python, no HTTP
- `wifi_exec("airmon_ng", data)` dispatch table

### Phase 2 — 12 *_direct.py Modules (115 tools total)

```bash
mcp_core/
├── wifi_direct.py              # 12 tools — airmon, airodump, aireplay, aircrack...
├── web_recon_direct.py         # 9 tools  — katana, gau, httpx, arjun, paramspider...
├── misc_direct.py              # 34 tools — ropgadget, volatility, ghidra, responder...
├── security_direct.py          # 11 tools — prowler, trivy, docker-bench, kube-hunter...
├── password_cracking_direct.py # 8 tools  — hydra, hashcat, john, hashid, medusa...
├── web_scan_direct.py          # 7 tools  — nikto, sqlmap, wpscan, dalfox, jaeles...
├── web_fuzz_direct.py          # 7 tools  — gobuster, ffuf, feroxbuster, dirb, wfuzz...
├── smb_enum_direct.py          # 6 tools  — enum4linux, netexec, rpcclient, smbmap...
├── net_scan_direct.py          # 5 tools  — nmap, masscan, rustscan, arp-scan
├── exploit_framework_direct.py # 5 tools  — metasploit, msfvenom, pwntools, pwninit...
├── recon_direct.py             # 7 tools  — amass, subfinder, autorecon, fierce, whois...
└── osint_direct.py             # 4 tools  — sherlock, spiderfoot, sublist3r, parsero
```
**Total: 115 tools across 12 modules** (Phase 2 complete)

### Phase 2 — gateway.py DIRECT_ROUTES

- `run_tool("hashid", {...})` → `pwdcrack_exec()` → `execute_command()` direct
- No Flask, no HTTP — full bypass
- Fallback to Flask for unmigrated tools only

---

## 🏗️ Architecture

### Before (legacy)

```markdown
mcp_tool → hexstrike_client.safe_post() → HTTP → Flask → server_api/ → execute_command()
```

### After (Phase 2)

```markdown
mcp_tool → *_direct.py → execute_command()  (direct Python call)
gateway run_tool → DIRECT_ROUTES → *_direct.py → execute_command()
```

### Target (Phase 3)

```bash
mcp_tool → *_direct.py → execute_command()
Flask removed → mcp.run(transport="http")
Agents → @mcp.prompt() native
```

---

## 🔄 Phase 3 — TODO

### 1. Remove Flask

```python
# Replace hexstrike_server.py with:
mcp.run(transport="http", host="127.0.0.1", port=8888)
```

### 2. Migrate Agents to mcp_core/

12 agents in `server_core/intelligence/` and `server_core/workflows/`:

- `IntelligentDecisionEngine` — tool selection + param optimization
- `BugBountyWorkflowManager` — bug bounty orchestration
- `CTFWorkflowManager` — CTF solving
- `CVEIntelligenceManager` — CVE intelligence
- `AIExploitGenerator` — exploit development
- `VulnerabilityCorrelator` — attack chain discovery
- `TechnologyDetector` — tech stack identification
- `RateLimitDetector` — rate limiting detection
- `FailureRecoverySystem` — error handling
- `PerformanceMonitor` — system optimization
- `ParameterOptimizer` — contextual param optimization
- `GracefulDegradation` — fault tolerance

### 3. FastMCP Prompts — native MCP workflows

```python
@mcp.prompt()
async def bug_bounty_workflow(target: str) -> list:
    workflow = BugBountyWorkflowManager()
    return workflow.create_reconnaissance_workflow(target)

@mcp.prompt()
async def wifi_attack_chain(interface: str, bssid: str) -> list:
    """Full chain: airmon → airodump → aireplay → aircrack"""
```

### 4. User Elicitation — confirm destructive actions

```python
result = await ctx.elicit(
    "Confirm deauth attack on AA:BB:CC:DD:EE:FF?",
    response_type=bool
)
if result.action == "accept" and result.data:
    # run aireplay_ng
```

**Note:** Supported in Claude Desktop ✅

### 5. Resources MCP

```python
@mcp.resource("scan://{target}/results")
async def scan_result(target: str) -> str:
    return get_cached_scan(target)

@mcp.resource("health://server")
async def server_health() -> str:
    return get_health_json()
```

### 6. Still on Flask — needs Phase 3

```bash
ops/ (8)            — complex system utilities
web_framework/ (2)  — browser_agent + http_framework
gateway.py          — central orchestrator
ai_assist/          — IntelligentDecisionEngine
bugbounty_workflow/ — BugBountyWorkflowManager
error_handling/     — internal utilities
ai_payload/         — payload generation
web_scan/burpsuite  — depends on browser_agent
```

---

## 🔬 FastMCP 3.x Features — Status

| Feature | Status | Notes |
|---|---|---|
| `ctx.info()` / `ctx.error()` | ✅ 70+ tools | Streams to LLM in real-time |
| `ctx.report_progress()` | ✅ 15-20 tools | Web fuzzing, scanning modules |
| `run_in_executor` | ✅ All tools | Non-blocking async I/O |
| `BM25SearchTransform` | ✅ server_setup.py | Tool discovery |
| `SkillsDirectoryProvider` | ✅ server_setup.py | Skills as MCP resources |
| `@mcp.tool()` decorator | ✅ 115 tools | FastMCP 3.x native |
| `@mcp.prompt()` | 🔄 Phase 3 | Workflows + LLM sampling |
| Resources MCP | 🔄 Phase 3 | Scan results via `@mcp.resource()` |
| User Elicitation | 🔄 Phase 3 | Destructive actions via `ctx.elicit()` |
| Dependency Injection | ✅ Supported | `ctx: Context = CurrentContext()` |
| FastMCP Apps | ❌ Low priority | prefab-ui too immature |

---

## 🧠 Key Patterns — FastMCP 3.x Aligned

### Tool Pattern (FastMCP 3.x — Modern)

**Recommended: Dependency Injection (v2.14+)**
```python
from fastmcp import FastMCP, Context
from fastmcp.dependencies import CurrentContext
import asyncio
import mcp_core.wifi_direct as _wifi_direct  # module-level — patchable

mcp = FastMCP(name="HexStrike")

@mcp.tool(description="Start WiFi monitor mode")
async def airmon_ng(
    ctx: Context = CurrentContext(),
    interface: str = "wlan0",
    action: str = "start"
) -> Dict[str, Any]:
    """Start/stop WiFi interface monitor mode."""
    await ctx.info(f"🔍 {action} monitor on {interface}")
    await ctx.report_progress(0, 100)

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None, lambda: _wifi_direct.wifi_exec("airmon_ng", {"interface": interface, "action": action})
    )

    # Non-blocking progress polling
    phases = [(25, "Phase 1..."), (50, "Phase 2..."), (75, "Phase 3...")]
    for progress, msg in phases:
        done, _ = await asyncio.wait([future], timeout=15)
        if done: break
        await ctx.report_progress(progress, 100)
        await ctx.info(msg)

    result = await future
    await ctx.report_progress(100, 100)
    
    if result.get("success"):
        await ctx.info("✅ Done")
    else:
        await ctx.error(f"❌ Failed: {result.get('error')}")
    return result
```

**Legacy: Type-Hint Injection (still supported)**
```python
@mcp.tool()
async def tool_name(ctx: Context, target: str) -> Dict[str, Any]:
    # Context auto-injected via type hint
    await ctx.info(f"Starting on {target}")
    return {"success": True}
```

### *_direct.py Pattern (Backend executor)

**Location:** `mcp_core/{module}_direct.py` — Pure Python, no Flask

```python
# mcp_core/wifi_direct.py
from server_core.command_executor import execute_command

def _require(data: dict, *keys) -> dict:
    """Validate required parameters."""
    for key in keys:
        if not data.get(key, ""):
            return {"success": False, "error": f"Missing: {key}"}
    return {}

def _airmon_ng(data: dict) -> dict:
    """Execute airmon-ng command."""
    err = _require(data, "interface", "action")
    if err: return err
    
    interface = data["interface"]
    action = data["action"]
    command = f"sudo airmon-ng {action} {interface}"
    return execute_command(command)

_HANDLERS = {
    "airmon_ng": _airmon_ng,
    "airodump_ng": _airodump_ng,
    # ... more handlers
}

def wifi_exec(tool: str, data: dict) -> dict:
    """Route tool request to handler."""
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown tool: {tool}"}
    return handler(data)
```

**Gateway routing** (`mcp_tools/gateway.py`): Maps tool names → direct executors
```python
DIRECT_ROUTES = {
    "airmon_ng": (wifi_exec, "airmon_ng"),
    "nmap": (net_scan_exec, "nmap"),
    "sherlock": (osint_exec, "sherlock"),  # osint_direct.py
    # ... 110+ tools directly routed
}
```

### Import Pattern for Test Patching

```python
# ✅ CORRECT: Module-level import — patchable in tests
import mcp_core.wifi_direct as _wifi_direct
# Later: _wifi_direct.wifi_exec("tool", data)

# ❌ WRONG: From import — NOT patchable in tests
from mcp_core.wifi_direct import wifi_exec
# Cannot mock: patch("wifi_exec") has no effect

# In tests — patch the module reference
with patch("mcp_core.wifi_direct.wifi_exec") as mock:
    mock.return_value = {"success": True}
    # Call tool, wifi_exec will be mocked
```

---

## 🧪 Testing — 200+ Test Functions, 48 Test Classes

```bash
# All tests in venv pytest
hexstrike-env/bin/pytest tests/ -v

# WiFi tests (37 functions, 12 classes)
hexstrike-env/bin/pytest tests/test_wifi_mcp_tools.py -v

# Endpoint tests (48 classes: system, net, web, crypto, cloud, k8s, forensics...)
hexstrike-env/bin/pytest tests/test_endpoints_exist.py -v
```

**Test class breakdown:**
- System ops, files, processes, wordlists, database (8 classes)
- Network: scanning, SMB, DNS, lookup (4 classes)
- Web: fuzzing, scanning, crawling, probing, framework (5 classes)
- Security: password cracking, exploitation, binary analysis (3 classes)
- Cloud/Container: audit, exploit, container, k8s (4 classes)
- Forensics: credential harvest, binary analysis, crypto, data (4 classes)
- WiFi pentest (12 classes, 37 test functions)
- Intelligence: workflows, errors, payloads (3 classes)

**Patch pattern for mocking:**
```python
with patch("mcp_core.wifi_direct.wifi_exec") as mock:
    mock.return_value = {"success": True, "output": "..."}
    tool = await mcp.get_tool("airmon_ng")  # async in FastMCP 3.x
    result = await tool.fn(ctx, interface="wlan0", action="start")
```

---

## 🌿 Branch Structure

```bash
refactor/fastmcp-modernization   ← active development
backup/phase2-complete-validated ← stable restore point (a909a02)
feat/wifi-pentest-clean          ← history — merged by upstream ✅
```

### Useful git commands

```bash
# Check upstream changes before rebase
git fetch upstream
git diff HEAD..upstream/beta/1.0.13 --name-only | grep "\.py$"

# Rebase on upstream
git rebase upstream/beta/1.0.13
git diff --name-only --diff-filter=U | xargs -r git checkout --ours
GIT_EDITOR=true git rebase --continue
git push --force-with-lease origin refactor/fastmcp-modernization

# Restore to Phase 2 backup
git checkout backup/phase2-complete-validated
```

---

## ⚠️ Known Issues / Notes

| Issue | Solution |
|---|---|
| `safe_postsafe_post` doublon | `content.replace("safe_postsafe_post", "safe_post")` then `re.sub` |
| `GIT_EDITOR=true` | Avoids vim on rebase --continue |
| ANSI colors in output | `HexStrikeColors` breaks JSON parsing in IDE — remove in Phase 3 |
| Flask Slowloris CVE-2007-6750 | Port 8888 Werkzeug dev server — fix in Phase 3 |
| `wsl.exe` in Claude Desktop | Use `cmd.exe /c hexstrike_mcp.bat` wrapper |
| `run_tool` gateway fallback | If tool not in DIRECT_ROUTES → still uses Flask |
| CAP_NET_RAW in WSL | nmap needs `-sT -Pn` flags — no raw socket in WSL |

---

## 🌍 Upstream Analysis (CommonHuman-Lab)

**Development approach:**
- Uses `logger.info(HexStrikeColors.X)` → terminal only, NOT LLM-aware
- Active on `ui/` React dashboard (Vite/Recharts)
- Added `active_directory/` module (7 new AD tools: adidnsdump, bloodhound, certipy...)
- Modified `*_direct.py` in recent commits → **check diffs before rebase**
- Legacy logging, no `ctx: Context` integration
- Blocking tool calls (no `run_in_executor`)
- Skills without FastMCP 3.x context streaming

**Our Key Differentiators (HexStrike VhaTer):**

1. ✅ **Context Streaming** → `ctx.info()/error()` → LLM sees real-time logs
2. ✅ **Direct Execution** → 12 `*_direct.py` modules → ~150 Flask HTTP calls eliminated
3. ✅ **Async Non-Blocking** → `run_in_executor` + `asyncio.wait()` → responsive tools
4. ✅ **Gateway Bypass** → `DIRECT_ROUTES` → 110 tools routed directly
5. ✅ **Comprehensive Tests** → 200+ test functions, 48 classes (not just 37)
6. ✅ **FastMCP 3.x Native** → Dependency injection, prompts, resources ready for Phase 3
7. ✅ **OSINT Integration** → `osint_direct.py` with sherlock, spiderfoot, etc.

---

## 🚀 Production Setup

### Antigravity IDE config

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

### Start server

```bash
cd ~/hexstrike-ai-community-edition
kill $(lsof -t -i:8888) 2>/dev/null
hexstrike-env/bin/python3 hexstrike_server.py > /tmp/hexstrike.log 2>&1 &
sleep 2 && curl -s http://127.0.0.1:8888/health | grep "all_essential"
```

### Validate Phase 2 direct execution

```bash
hexstrike-env/bin/python3 - << 'EOF'
import sys; sys.path.insert(0, '.')
from mcp_core.password_cracking_direct import pwdcrack_exec
result = pwdcrack_exec("hashid", {
    "hash_value": "5f4dcc3b5aa765d61d8327deb882cf99",
    "additional_args": "-m"
})
print("✅ MD5 mode 0" if "MD5" in result.get("output","") else "❌ FAIL")
EOF
```

---

## 📚 FastMCP 3.x Resources

- Context API: <https://gofastmcp.com/servers/context>
- Tool Search: <https://gofastmcp.com/servers/transforms/tool-search>
- Client: <https://gofastmcp.com/clients/basic-usage>
- HTTP Transport: <https://gofastmcp.com/deployment/running-server>
- FileSystemProvider: <https://gofastmcp.com/servers/providers/filesystem>
- Prompts: <https://gofastmcp.com/servers/prompts>
- Resources: <https://gofastmcp.com/servers/resources>

---

## 🎯 Next Session Priorities

1. **Check upstream diff** on `*_direct.py` — what did they change?
2. **Rebase** on `upstream/beta/1.0.13` — get `active_directory/` module
3. **Add `active_directory/` to `misc_direct.py`** — 7 new tools
4. **Phase 3** — User Elicitation on destructive tools (aireplay_ng, responder, metasploit)
5. **Resources MCP** — expose health + scan results
6. **PR** toward CommonHuman-Lab
