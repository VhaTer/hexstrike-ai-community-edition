# AGENTS.md

## Guidance for agentic coding assistants working in this repository

## Quick orientation

- **Project:** HexStrike AI Community Edition — FastMCP 3.x (Python 3.8+)
- **Primary entrypoint:** `hexstrike_server.py` (Phase 3 standalone MCP server — no Flask)
- **Phase:** Phase 2 complete (115 tools, 12 `*_direct.py` modules) + Phase 3 server active
- **Config:** `config.py`, `server_core/config_core.py`, use `config_core.get(key, default)`
- **Tools registry:** `tool_registry.py` + gateway routing in `mcp_tools/gateway.py`
- **Tests:** 37 test functions, 12 test classes — wifi_pentest suite (37/37 passing ✅)

## Build, lint, test commands

### Environment setup

```bash
cd ~/hexstrike-ai-community-edition
source hexstrike-env/bin/activate
# or use full path: hexstrike-env/bin/python3
```

### Start the Phase 3 server (no Flask)

```bash
cd ~/hexstrike-ai-community-edition
kill $(lsof -t -i:8888) 2>/dev/null
hexstrike-env/bin/python3 hexstrike_server.py
# Server: http://127.0.0.1:8888 — FastMCP native HTTP/SSE transport
```

### Start the MCP client (connects to running server)

```bash
hexstrike-env/bin/python3 hexstrike_mcp.py --server http://127.0.0.1:8888 --profile full
```

### Tests

```bash
# WiFi penetration testing (37 functions, 12 classes) — always run this
hexstrike-env/bin/pytest tests/test_wifi_mcp_tools.py -v

# Single test function
hexstrike-env/bin/pytest tests/test_wifi_mcp_tools.py::TestAirmonNg::test_tool_registered
```

**Key test patterns:**
- Always use venv pytest: `hexstrike-env/bin/pytest` (not system pytest)
- Patch module-level imports for mocking: `patch("mcp_core.wifi_direct.wifi_exec")`
- Mock context: `ctx = MagicMock()`, `ctx.info = AsyncMock()`, `ctx.report_progress = AsyncMock()`
- 37/37 WiFi tests passing ✅

### Validate direct execution (no server needed)

```bash
hexstrike-env/bin/python3 -c "import mcp_core.osint_direct; print('✅ OK')"

hexstrike-env/bin/python3 - << 'EOF'
import sys; sys.path.insert(0, '.')
from mcp_core.password_cracking_direct import pwdcrack_exec
result = pwdcrack_exec("hashid", {"hash_value": "5f4dcc3b5aa765d61d8327deb882cf99", "additional_args": "-m"})
print("✅ MD5" if "MD5" in result.get("output","") else "❌ FAIL")
EOF
```

## Architecture

### Phase 3 server flow (current)

```
Claude / MCP client
    ↓ MCP protocol
hexstrike_server.py → mcp.run(transport="http", port=8888)
    ↓
server_setup.setup_mcp_server_standalone()
    ↓ run_security_tool("nmap", '{"target": "..."}')
mcp_core/*_direct.py → execute_command()
    ↓
Shell execution → result dict
```

### Legacy flow (hexstrike_mcp.py — still used for profile-based loading)

```
hexstrike_mcp.py → HexStrikeClient → safe_post() → DIRECT_ROUTES bypass → *_direct.py
                                                  → Flask fallback (unmigrated tools)
```

## Code style guidelines

### Imports

- Module-level imports for `*_direct.py` — required for test patching:
  ```python
  # ✅ CORRECT — patchable
  import mcp_core.wifi_direct as _wifi_direct

  # ❌ WRONG — not patchable in tests
  from mcp_core.wifi_direct import wifi_exec
  ```

### FastMCP 3.x Context — correct pattern

```python
from fastmcp import Context
import asyncio
import mcp_core.wifi_direct as _wifi_direct  # module-level import

@mcp.tool()
async def airmon_ng(ctx: Context, interface: str, action: str = "start") -> Dict[str, Any]:
    """Start/stop WiFi interface monitor mode."""
    data = {"interface": interface, "action": action}

    await ctx.info(f"🔍 {action} monitor on {interface}")
    await ctx.report_progress(0, 100)

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None, lambda: _wifi_direct.wifi_exec("airmon_ng", data)
    )

    phases = [(25, "Phase 1..."), (50, "Phase 2..."), (75, "Phase 3...")]
    for progress, msg in phases:
        done, _ = await asyncio.wait([future], timeout=15)
        if done:
            break
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

**Note:** `ctx: Context` is injected automatically by FastMCP via type hint.
Do NOT use `ctx: Context = CurrentContext()` — not available in FastMCP 3.x standard.

### *_direct.py pattern (backend executor)

```python
# mcp_core/example_direct.py
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

### Error returns

Always return `{"success": False, "error": "message"}` — never raise exceptions from tool handlers.

### Naming

- Modules/files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## *_direct.py modules (Phase 2 — 12 modules, 115 tools)

```
mcp_core/
├── wifi_direct.py              # 12 tools — airmon, airodump, aireplay, aircrack...
├── recon_direct.py             # 7 tools  — amass, subfinder, theharvester...
├── net_scan_direct.py          # 5 tools  — nmap, masscan, rustscan, arp-scan
├── web_scan_direct.py          # 7 tools  — nikto, sqlmap, wpscan, dalfox...
├── web_fuzz_direct.py          # 7 tools  — gobuster, ffuf, feroxbuster...
├── password_cracking_direct.py # 8 tools  — hydra, hashcat, john, hashid...
├── smb_enum_direct.py          # 6 tools  — enum4linux, netexec, smbmap...
├── exploit_framework_direct.py # 5 tools  — metasploit, msfvenom, pwntools...
├── web_recon_direct.py         # 9 tools  — katana, gau, httpx, arjun...
├── security_direct.py          # 11 tools — prowler, trivy, kube-hunter...
├── misc_direct.py              # 28 tools — binary, forensics, db, api_scan...
└── osint_direct.py             # 4 tools  — sherlock, spiderfoot, sublist3r, parsero
```

Adding a new tool: add handler to the appropriate `*_direct.py` + register in `gateway.py` DIRECT_ROUTES + add to `server_setup.py` DIRECT_TOOLS.

## Known issues / Notes

| Issue | Solution |
|---|---|
| `testing.py` deleted by Windows Defender | Already removed from `singletons.py` imports |
| `GIT_EDITOR=true` | Avoids vim on `git rebase --continue` |
| ANSI colors in output | `HexStrikeColors` breaks JSON — remove in Phase 3 cleanup |
| `wsl.exe` in Claude Desktop | Use `cmd.exe /c hexstrike_mcp.bat` wrapper |
| CAP_NET_RAW in WSL | nmap needs `-sT -Pn` flags — no raw socket in WSL |
| Stash before rebase | `git stash push -u -m "label"` — session .md files are untracked |

## Repo-specific notes

- **Upstream:** `upstream/beta/rootkitfox` — rebased 2026-03-27
- **Our branch:** `refactor/fastmcp-modernization` HEAD `678511b`
- **Backup:** `backup/phase2-complete-validated` — safe restore point
- **Server binds to** `127.0.0.1` by default; override via `HEXSTRIKE_HOST`
- **Default port:** `8888`; override via `HEXSTRIKE_PORT`
