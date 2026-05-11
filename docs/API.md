# HexStrike AI-PULSE — API Documentation

## Architecture

```
CLI (hexstrike.py) ──→ DIRECT_ROUTES ──→ *_direct.py ──→ subprocess
                          (128 routes)       (16 modules)

MCP Client ──→ server_setup.py ──→ DIRECT_TOOLS ──→ *_direct.py ──→ subprocess
               (FastMCP 3)         (130 entries)

HTTP/SSE ──→ hexstrike_server.py ──→ DIRECT_TOOLS ──→ *_direct.py
```

**Key principle**: No Flask. Every tool executes directly through a `*_direct.py` module.

## Tool Registration

Tools are registered in `mcp_core/server_setup.py` via the `DIRECT_TOOLS` dict:

```python
DIRECT_TOOLS = {
    "nmap":     (net_scan_exec, "nmap"),
    "sqlmap":   (web_scan_exec, "sqlmap"),
    "airmon_ng":(wifi_exec, "airmon_ng"),
    ...
}
```

Each entry maps a tool name to `(executor_function, tool_name)`. The executor is a dispatch function in one of the 16 `*_direct.py` modules.

## Tool Schemas

Tool parameter metadata is defined in `tool_registry.py`:

```python
TOOLS = {
    "nmap": {
        "desc": "Port scan and service detection",
        "category": "network_recon",
        "params": {"target": {"required": True}},
        "optional": {"scan_type": "-sCV", "ports": "", "additional_args": "-T4 -Pn"},
        "effectiveness": 0.95,
    },
}
```

Only `get_tool(name)` is used at runtime — it's a lookup dict, not a registration system.

## Direct Execution Modules

| Module | Dispatch fn | Tools | Category |
|--------|-------------|-------|----------|
| `net_scan_direct.py` | `net_scan_exec` | nmap, masscan, rustscan | Network scan |
| `web_scan_direct.py` | `web_scan_exec` | sqlmap, nikto, wpscan, xsser, dalfox, jaeles, commix, zap | Web vuln scan |
| `web_fuzz_direct.py` | `web_fuzz_exec` | ffuf, gobuster, dirb, dirsearch, feroxbuster, wfuzz, dotdotpwn | Web fuzzing |
| `web_probe_direct.py` | `web_probe_exec` | whatweb, httpx, wafw00f | Web probe |
| `wifi_direct.py` | `wifi_exec` | airmon-ng, airodump-ng, aireplay-ng, aircrack-ng, ... | WiFi pentest |
| `osint_direct.py` | `osint_exec` | sherlock, spiderfoot, sublist3r, parsero | OSINT |
| `active_directory_direct.py` | `ad_exec` | impacket, ldapdomaindump, bloodhound, mitm6, ... | AD |
| `smb_enum_direct.py` | `smb_enum_exec` | smbmap, enum4linux, nbtscan, nxc, rpcclient | SMB |
| `recon_direct.py` | `recon_exec` | amass, subfinder, dnsenum, fierce, theharvester, ... | Recon |
| `password_cracking_direct.py` | `pwd_crack_exec` | john, hashcat, hydra, medusa, patator, ophcrack | Cracking |
| `exploit_framework_direct.py` | `exploit_exec` | metasploit, msfvenom, searchsploit | Exploitation |
| `misc_direct.py` | `misc_exec` | nuclei, testssl, bbot, scalpel, bulk_extractor, ... | Misc |
| `security_direct.py` | `security_exec` | checksec, pwntools, ropper, ropgadget, gdb, strings, ... | Binary |
| `testssl_direct.py` | `testssl_exec` | testssl | SSL/TLS |
| `vuln_intel_direct.py` | `vuln_intel_exec` | vulnx | Vuln intel |
| `ctf_engine.py` | CTF tools | ctf_analyze, ctf_solve, ctf_tools, ctf_team | CTF |

## Entrypoints

### 1. Unified CLI (`hexstrike.py`)

```bash
python3 hexstrike.py serve           # Start MCP server
python3 hexstrike.py scan nmap --target 10.0.0.1
python3 hexstrike.py tools           # List tools
python3 hexstrike.py status          # Environment status
python3 hexstrike.py validate        # Validate environment
python3 hexstrike.py mcp             # Run MCP stdio
python3 hexstrike.py ctf pwn/hard    # CTF workflow plan
```

Flags: `--json` (structured output), `-o FILE` (write output).

### 2. HTTP/SSE Server (`hexstrike_server.py`)

```
http://127.0.0.1:8888/mcp           # SSE endpoint
http://127.0.0.1:8888/dashboard     # Web UI dashboard (requires `ui/` build)
```

### 3. MCP stdio (`hexstrike_mcp.py`)

```
python3 hexstrike_mcp.py             # stdio MCP (for Claude Desktop)
```

Config: `hexstrike-ai-mcp.json` (example MCP client config).

## Context

All tool functions accept `ctx: Context` (FastMCP) for progress reporting:

```python
await ctx.report_progress(50, 100)
await ctx.info("Scanning...")
await ctx.error("Failed")
```

Destructive tools use `confirm_destructive_action(ctx, action, detail, warning)` via `mcp_core/elicitation.py`.

## Caching

Scan results are cached per session with adaptive TTL (30-90 min based on execution time). Cache key: `{session_id}:{tool}:{target}:[hash]`. Max 500 entries, LRU eviction.

## Environment Overrides

- `HEXSTRIKE_HOST`, `HEXSTRIKE_PORT` — server address
- `HEXSTRIKE_DATA_DIR` — data directory
- `HEXSTRIKE_API_KEY` — API key for HTTP auth
