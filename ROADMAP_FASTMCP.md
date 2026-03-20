# HexStrike CE — FastMCP 3.x Migration Roadmap

**Branch:** `refactor/fastmcp-modernization`  
**Author:** VhaTer  
**Base:** `upstream/beta/1.0.13`  
**FastMCP target:** `>=3.1.0`

---

## Why this roadmap exists

HexStrike CE was originally built on top of HexStrike V6 — a Flask-based architecture
where `hexstrike_server.py` exposes ~180 tool endpoints over HTTP, and `mcp_tools/`
calls them via `hexstrike_client.safe_post()`.

FastMCP 3.x changed the game. It is no longer just an MCP wrapper — it is a full
async server framework with native HTTP transport, a Client system, Resources,
Prompts, and real-time Context streaming. The Flask layer is now redundant and
actively limits what the LLM can see and do.

This roadmap proposes a clean, non-breaking 3-phase migration.

---

## What FastMCP 3.x brings that matters for this project

### 1. `Context` — real-time LLM feedback (already partially done ✅)

```python
# Before (upstream current state)
logger.info(f"{HexStrikeColors.FIRE_RED}🔍 Scanning {target}{HexStrikeColors.RESET}")
# → logs to terminal only, LLM sees nothing during execution

# After (our branch)
await ctx.info(f"🔍 Starting nmap scan: {target}")
await ctx.report_progress(50, 100)
await ctx.error(f"❌ nmap failed: {result['error']}")
# → streams directly to the LLM via MCP protocol
```

`HexStrikeColors` ANSI codes are meaningless in MCP context — the LLM receives
plain text. `ctx` replaces the entire logger+colors pattern for MCP tools.

### 2. `BM25SearchTransform` — on-demand tool discovery (already done ✅)

```python
transforms = [BM25SearchTransform()]
mcp = FastMCP("hexstrike-ai-mcp", transforms=transforms)
```

With 180+ tools, loading everything at once overwhelms the LLM context window.
BM25SearchTransform exposes only the tools semantically relevant to the current
query — critical for a project of this scale.

### 3. FastMCP `Client` — removes Flask dependency (Phase 1)

```python
# Before: every tool does this
result = await loop.run_in_executor(
    None, lambda: hexstrike_client.safe_post("api/tools/nmap", data)
)
# → HTTP round-trip to Flask → subprocess → response

# After (Phase 1): direct in-process call
from fastmcp import Client
async with Client(mcp_server) as client:
    result = await client.call_tool("nmap_scan", {"target": target})
# → direct Python call, no HTTP, no Flask needed
```

### 4. Native HTTP/SSE transport — replaces Flask (Phase 3)

```python
# hexstrike_server.py Flask app → replaced by:
mcp.run(transport="http", host="127.0.0.1", port=8888)
```

FastMCP 3.1+ natively exposes an HTTP/SSE endpoint. `hexstrike_server.py`
(currently ~8000 lines) becomes unnecessary.

### 5. `Resources` and `Prompts` — workflow intelligence (future)

```python
@mcp.resource("scan://{target}")
async def scan_resource(target: str) -> str:
    """Expose scan results as MCP resources."""
    ...

@mcp.prompt()
async def pentest_workflow(target: str, scope: str) -> list:
    """Guide the LLM through a full pentest workflow."""
    return [
        {"role": "user", "content": f"Target: {target}, Scope: {scope}"},
        {"role": "assistant", "content": "Starting recon phase..."},
    ]
```

This enables guided bug bounty and CTF workflows directly in the LLM — 
complementing the existing `.opencode/agents/` system.

---

## Current state of `refactor/fastmcp-modernization`

### Done (22 commits ahead of beta/1.0.13)

| What | Files | Status |
|------|-------|--------|
| Simplify MCP startup for FastMCP 3.1+ | `mcp_core/mcp_entry.py`, `server_setup.py` | ✅ |
| `mcp.run(show_banner=False, log_level="WARNING")` | `mcp_core/mcp_entry.py` | ✅ |
| `BM25SearchTransform` for tool discovery | `mcp_core/server_setup.py` | ✅ |
| `SkillsDirectoryProvider` for skills/ | `mcp_core/server_setup.py` | ✅ |
| `ctx: Context` migration — net_scan | `nmap, masscan, rustscan, arp_scan` | ✅ |
| `ctx: Context` migration — recon | `amass, subfinder, autorecon, theharvester` | ✅ |
| `ctx: Context` migration — web_fuzz | `gobuster, ffuf, feroxbuster, dirb, dirsearch, wfuzz, dotdotpwn` | ✅ |
| `ctx: Context` migration — web_scan | `nikto, sqlmap, dalfox, jaeles, wpscan, burpsuite, xsser, zap` | ✅ |
| `ctx: Context` migration — web_crawl | `katana, hakrawler` | ✅ |
| `ctx: Context` migration — web_probe | `httpx` | ✅ |
| `ctx: Context` migration — url_recon | `gau, waybackurls` | ✅ |
| `ctx: Context` migration — param_discovery | `arjun, paramspider, x8` | ✅ |
| `ctx: Context` migration — smb_enum | `enum4linux, nbtscan, netexec, rpcclient, smbmap` | ✅ |
| `ctx: Context` migration — password_cracking | `hydra, hashcat, john, medusa, patator, ophcrack, hashid` | ✅ |
| `ctx: Context` migration — vuln_scan | `nuclei` | ✅ |
| `ctx: Context` migration — credential_harvest | `responder` | ✅ |
| `ctx: Context` migration — memory_forensics | `volatility, volatility3` | ✅ |
| `ctx: Context` migration — exploit_framework | `metasploit, msfvenom, pwninit, pwntools, exploit_db` | ✅ |
| `ctx: Context` migration — error_handling | `statistics, test_recovery` | ✅ |
| `ctx: Context` migration — ops | `auto_install, process_management, system_monitoring, visual_output` | ✅ |
| `ctx: Context` migration — dns_enum | `dnsenum, fierce` | ✅ |
| `ctx: Context` migration — net_lookup | `whois` | ✅ |
| WiFi pentest docstrings — workflow context | all 12 wifi tools | ✅ |
| Stub `FileUploadTestingFramework` | `server_core/workflows/bugbounty/testing.py` | ✅ |

### In progress

| What | Status |
|------|--------|
| `ctx: Context` migration — wifi_pentest | ⚠️ lost during rebase, needs redo |
| Unit tests — wifi_pentest mcp_tools | 🔄 in progress |

### Known divergence from upstream

Upstream `beta/1.0.13` added `run_in_executor` to all tools (correct async pattern).
Our branch has `ctx: Context` but some tools lost `run_in_executor` during rebase
conflicts. Both are needed — the correct pattern is:

```python
@mcp.tool()
async def nmap_scan(ctx: Context, target: str, ...) -> Dict[str, Any]:
    await ctx.info(f"🔍 Starting nmap scan: {target}")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: hexstrike_client.safe_post("api/tools/nmap", data)
    )
    if result.get("success"):
        await ctx.info(f"✅ nmap completed for {target}")
    else:
        await ctx.error(f"❌ nmap failed: {result.get('error', 'unknown')}")
    return result
```

`run_in_executor` keeps the asyncio event loop responsive.
`ctx` streams progress to the LLM in real time.
Both are required. Neither replaces the other.

---

## Phase 1 — FastMCP Client for new modules (next)

**Goal:** New modules bypass Flask entirely. Existing modules untouched.

**Target module:** `wifi_pentest` (no existing Flask blueprint conflict)

```
Before:
  mcp_tools/wifi_pentest/airmon_ng.py
    → hexstrike_client.safe_post("api/tools/wifi_pentest/airmon_ng", data)
    → HTTP → Flask → server_api/wifi_pentest/airmon_ng.py
    → subprocess

After (Phase 1):
  mcp_tools/wifi_pentest/airmon_ng.py
    → FastMCP Client → server_api/wifi_pentest/airmon_ng.py
    → subprocess (direct, no HTTP)
```

**Implementation sketch:**

```python
# mcp_core/direct_client.py — new file
from fastmcp import Client
from mcp_core.server_setup import setup_mcp_server

_direct_mcp = None

async def get_direct_client():
    global _direct_mcp
    if _direct_mcp is None:
        # Mount server_api tools directly, no Flask
        from server_api.wifi_pentest import register_wifi_blueprint_as_mcp
        _direct_mcp = FastMCP("hexstrike-direct")
        register_wifi_blueprint_as_mcp(_direct_mcp)
    return Client(_direct_mcp)
```

**Why wifi_pentest first:**
- Cleanest module — added in beta/1.0.12, no V6 legacy
- All tools follow same pattern
- No shared state with Flask blueprints
- Already has `run_in_executor` from upstream

---

## Phase 2 — Replace Flask blueprints one by one

**Goal:** Each `server_api/` blueprint gets a FastMCP-native equivalent.
`hexstrike_client.safe_post()` calls are replaced module by module.

**Order of migration (suggested):**
1. `wifi_pentest` — Phase 1
2. `recon`, `dns_enum`, `net_lookup` — stateless, no side effects
3. `net_scan`, `web_probe`, `web_crawl` — stateless
4. `web_fuzz`, `web_scan`, `vuln_scan` — stateless
5. `password_cracking`, `exploit_framework` — needs careful testing
6. `ops`, `process_management` — last, most complex

---

## Phase 3 — Remove Flask

**Goal:** `hexstrike_server.py` is replaced by FastMCP native HTTP transport.

```python
# hexstrike_server.py becomes:
from mcp_core.server_setup import setup_mcp_server_standalone

mcp = setup_mcp_server_standalone()
mcp.run(transport="http", host="127.0.0.1", port=8888)
```

All `server_api/` blueprints become FastMCP tools registered directly.
Flask, `requests`, `hexstrike_client.py` are removed from dependencies.

---

## PR checklist (before opening)

- [ ] `ctx: Context` + `run_in_executor` on all migrated tools
- [ ] Unit tests passing — `hexstrike-env/bin/pytest tests/`
- [ ] `hexstrike_server.py` starts clean
- [ ] `hexstrike_mcp.py --profile wifi_pentest` connects and streams ctx logs
- [ ] `ROADMAP_FASTMCP.md` included in PR description

---

## References

- FastMCP Context: https://gofastmcp.com/servers/context
- FastMCP Tool Search: https://gofastmcp.com/servers/transforms/tool-search
- FastMCP Client: https://gofastmcp.com/clients/basic-usage
- FastMCP HTTP transport: https://gofastmcp.com/deployment/running-server
- FastMCP Resources: https://gofastmcp.com/servers/resources
- FastMCP Prompts: https://gofastmcp.com/servers/prompts
