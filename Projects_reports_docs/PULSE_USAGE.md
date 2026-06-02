# Pulse Usage Guide

HexStrike Pulse — 170+ security tools exposed as MCP primitives.

---

## Quick Start

```bash
# Start the Pulse HTTP server
./hexstrike-pulse

# In another terminal, connect your AI client to http://localhost:8888/mcp
```

---

## Slash Commands

Copy files from `hexstrike-commands/` to your client's commands directory.
See `hexstrike-commands/INSTALL.md` for per-client instructions.

| Command | What it does | Tool(s) called |
|---|---|---|
| `/scan <target> [intensity]` | Port scan + tech detection + vuln scan + attack plan | `scan()` — entry point |
| `/dashboard` | Full monitoring UI (18 panels) | `pulse_dashboard()` |
| `/recon <target>` | Surface + findings + plan in one | `get_surface()`, `get_findings()`, `get_plan()` |

### /scan

```
/scan 192.168.1.165 quick
/scan http://example.com full
/scan 10.0.0.0/24 medium
```

Intensity levels:

| Level | Tools | Time |
|---|---|---|
| quick | nmap + whatweb | ~30s |
| medium | + nuclei + nikto | ~2-3 min |
| full | + gobuster (web targets) | ~5-10 min |

### /dashboard

```
/dashboard
```

Opens the Prefab UI dashboard. No arguments needed. Covers: overview, scope,
surface, findings, plan, active tools, history, rate limit, errors & failures,
tool performance, cache status, intelligence, missing tools, async scans.

### /recon

```
/recon 192.168.1.165
```

Consolidated analysis: ports + services + technologies + vulnerabilities + attack
plan. Use after `/scan` has been run (leverages cached results).

---

## Adding Custom Commands

1. Copy `hexstrike-commands/TEMPLATE.md` to `<name>.md`.
2. Edit the frontmatter `description` (appears in TUI command list).
3. Write the prompt body — tell the LLM which Pulse tool to call and how.
4. Copy to your client's commands directory.
5. Restart your client.

**Convention**: a command should call exactly one Pulse tool, or a tightly
related chain (max 3). If you need more, it belongs in a new Pulse tool.

---

## Pulse MCP Tool Architecture

Pulse registers tools via FastMCP. The LLM discovers tools through
`search_tools(pattern)` and calls them via `call_tool(name, args)`.

### Entry points (always visible)

| Tool | Description |
|---|---|
| `scan()` | Reconnaissance scan, returns surface + findings + plan |
| `scan_background()` | Same but async, returns task_id immediately |
| `pulse_dashboard()` | Prefab UI dashboard |
| `get_live_dashboard()` | All 18+ panels as JSON |

### Discovery (via search_tools)

- `get_surface(target)` — open ports, services, technologies
- `get_findings(target)` — vulnerability findings, sorted by severity × exploit score
- `get_plan(target, objective)` — suggested attack chain with probability × ETA
- `get_history(target, limit)` — scope-filtered scan history
- `get_overview()` — version, uptime, RAM, tools count
- `get_active_tools()` — running processes
- `get_errors_and_failures()` — aggregate error statistics
- `get_tool_performance()` — per-tool success rate + timeouts
- `get_cache_status()` — cache hits/misses/ratio + per-tool breakdown
- `get_cache_intelligence()` — per-tool TTL, hit ratio, adaptive scores
- `http_request(url, ...)` — generic HTTP primitive (curl wrapper)
- `run_async_tool(tool, target, params)` — launch any tool in background

And 160+ typed wrappers for security tools (nmap, whatweb, sqlmap, gobuster,
nuclei, nikto, hydra, metasploit, etc.).

### Data sources

| Source | What |
|---|---|
| `_op_metrics` | Operational stats (errors, timeouts, cache hits) |
| `_scan_cache` | Scan results (TTL-adaptive, cross-session) |
| `get_tool_stats_store()` | Tool effectiveness (success rates, runs) |
| `get_target_store()` | Persistent findings per target (MCP resources) |
| `get_decision_engine()` | Attack planner (probabilities, ETAs) |
| `ToolRegistry` | 170 tools tracked, 90+ available on a typical system |
| `TelemetryPipeline` | Event buffer + per-tool aggregation |

### Targets

Most tools auto-detect the active target from the most recent `_scan_cache`
entry. Explicit target overrides auto-detection.

Target types: IP, domain, URL. Web targets (http:// or https://) get additional
treatment: hostname extraction for nmap, URL scanning for tools like whatweb.

---

## Server Commands

```bash
# Start Pulse HTTP server (background)
./hexstrike-pulse

# Start with debug logging
./hexstrike-pulse --debug

# Start on specific port
./hexstrike-pulse --port 9999

# Start HTTP mode (MCP over Streamable HTTP)
./hexstrike-pulse --transport http

# Stop (kill background process)
pkill -f hexstrike_mcp.py

# Run tests (from repo root)
source hexstrike-env/bin/activate
python -m pytest tests/ -m "not slow" -n auto -q
```

---

## Client Configuration

### opencode

```json
{
  "mcp": {
    "pulse": {
      "type": "remote",
      "url": "http://localhost:8888/mcp",
      "enabled": true
    }
  }
}
```

Set `timeout: 300000` in `opencode.json` for long scans.

### Claude Desktop

```json
{
  "mcpServers": {
    "pulse": {
      "command": "/path/to/hexstrike-pulse",
      "args": ["--transport", "stdio"],
      "env": {}
    }
  }
}
```

### Cline / Continue

Configure as a remote MCP server at `http://localhost:8888/mcp`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `connection refused` | Server not running | `./hexstrike-pulse` |
| Tools not found | Server not connected | Check MCP config |
| `stateless=True required` | opencode remote MCP | Already set in Pulse |
| Dashboard not rendering | prefab-ui <0.15.0 required | `pip install 'prefab-ui>=0.14.0,<0.15.0'` |
| Lock file error | Stale from crash | Auto-cleaned after 5s |
| Nikto timeout | Needs >300s | Scan intensity=medium without nikto |

---

## File Layout

```
hexstrike-ai-community-edition/
├── hexstrike_mcp.py            # MCP server entry point (stdio)
├── hexstrike_server.py         # HTTP server (dashboard + health)
├── hexstrike-pulse             # Launcher script
├── hexstrike.py                # CLI entry point
├── pulse_app.py                # Prefab UI dashboard
├── mcp_core/
│   ├── mcp_entry.py            # FastMCP server setup
│   ├── server_setup.py         # Tool registration
│   ├── tool_routes.py          # 170+ tool→binary mappings
│   ├── instructions.py         # System prompt (Couche 2)
│   ├── misc_direct.py          # Curl, strings, xxd, etc.
│   ├── binary_direct.py        # Varint, block parser, RLE, signal decode
│   └── wifi_direct.py          # Aircrack, tshark, etc.
├── hexstrike-commands/
│   ├── scan.md                 # /scan slash command
│   ├── dashboard.md            # /dashboard slash command
│   ├── recon.md                # /recon slash command
│   ├── TEMPLATE.md             # Template for new commands
│   └── INSTALL.md              # Per-client install instructions
└── tests/
    └── test_*.py               # 2869+ tests
```
