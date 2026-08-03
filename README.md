<p align="center">
  <img src="assets/hexstrike-pulse-logo.png" alt="HexStrike Pulse" width="400">
</p>

# HexStrike AI-PULSE

AI-powered security orchestration engine — 163 tools, live Prefab dashboards, any MCP client.

[![Python](https://img.shields.io/badge/Python-3.11%2B-red)]()
[![MCP](https://img.shields.io/badge/MCP-Streamable%20HTTP-red)]()
[![Coverage](https://img.shields.io/badge/Coverage-99%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-AGPLv3-green)]()
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/VhaTer/hexstrike-ai-community-edition)

---

## Quick start

```bash
git clone https://github.com/VhaTer/hexstrike-ai-community-edition.git
cd hexstrike-ai-community-edition
python3 -m venv hexstrike-env
source hexstrike-env/bin/activate
pip install -r requirements.txt
```

**Prerequisites:** Linux (Kali/Debian/Ubuntu), Python 3.11+. Security tools (`nmap`, `whatweb`, `nuclei`, `nikto`, `gobuster`, `sqlmap`, …) autodetected at runtime.

```bash
python3 hexstrike.py validate        # check tool availability
```

---

## Launch

```bash
./hexstrike-pulse                           # start HTTP server (background)
```

| Command | What it does |
|---------|-------------|
| `./hexstrike-pulse` | Start server on `:8888` (background) |
| `./hexstrike-pulse --foreground` | Start with visible logs |
| `./hexstrike-pulse stop` | Stop the server |
| `./hexstrike-pulse status` | Check if running |
| `./hexstrike-pulse --bridge` | Stdio bridge for Claude Desktop |

All clients share one server — no lock conflicts, no duplicate tools.

---

## Connect your AI agent

### Claude Desktop (WSL)

```json
{
  "mcpServers": {
    "hexstrike-pulse": {
      "command": "wsl.exe",
      "args": ["-d", "kali-linux", "/path/to/hexstrike-pulse", "--bridge"]
    }
  }
}
```

### OpenCode / Cline / any MCP client

```json
{
  "mcp": {
    "hexstrike-pulse": {
      "type": "remote",
      "url": "http://localhost:8888/mcp"
    }
  }
}
```

### Claude Code (terminal)

Server mode (MCP HTTP) — via CLI:

```bash
python3 hexstrike.py serve
# → http://127.0.0.1:8888/mcp
```

Or via the launcher:

```bash
./hexstrike-pulse --foreground
```

### Any MCP client (alternative)

Point your client to `http://localhost:8888/mcp` with type `streamable-http`.

---

## What you can do

### Orchestrate

```
> I'm a security researcher. Show me the attack surface of scanme.nmap.org.
```

The agent runs nmap, whatweb, nuclei, and nikto in sequence, resolves findings, and returns ports, services, technologies, vulnerabilities, and next steps. All results cached.

### Monitor

The Pulse dashboard is the Prefab UI opened via the `pulse_dashboard()` MCP tool while scans run. The same data is exposed as JSON on `http://127.0.0.1:8888/web-dashboard`. 3 zones:

| Tab | Panels |
|-----|--------|
| **Overview & Workflow** | Header, Scope, Surface, Plan IDE, Active Tools, Async Scans, System Trends (CPU/MEM chart), Cache Status, Cache Intelligence, Missing Tools, Rate Limit, Intelligence, Network I/O |
| **Findings** | Severity breakdown chart, Findings details |
| **History** | History, Sessions, Errors & Failures, Tool Performance, Confirmations |

### Execute

```bash
python3 hexstrike.py scan nmap scanme.nmap.org
python3 hexstrike.py scan whatweb http://example.com
python3 hexstrike.py scan nuclei http://example.com -p severity=critical
python3 hexstrike.py tools
```

### CLI Reference

#### `python3 hexstrike.py serve`

Start the Pulse HTTP/SSE server.

| Flag | Description | Default |
|------|-------------|---------|
| *(none)* | Start on `127.0.0.1:8888` | |
| `--host 0.0.0.0` | Bind address | `127.0.0.1` |
| `--port 8080` | Bind port | `8888` |
| `--debug` | Debug logging | off |

#### `python3 hexstrike.py scan <tool> [target]`

Run any security tool directly.

| Arg | Description |
|-----|-------------|
| `tool` | **Required.** Tool name (nmap, nuclei, whatweb, sqlmap, gobuster, …) |
| `target` | Optional. IP / URL / domain |

| Flag | Description |
|------|-------------|
| `-p key=val` | Extra parameter (repeatable) |
| `--json` | Raw JSON output |
| `-o file` | Write output to file |

**Examples:** `scan nmap 10.0.0.1`, `scan nmap 10.0.0.1 --json -o result.json`, `scan nuclei http://example.com -p severity=critical`, `scan nmap 10.0.0.1 -p scan_type=-sV -p ports=22,80`

#### `python3 hexstrike.py tools`

List available tools with descriptions.

| Flag | Description |
|------|-------------|
| `-f nmap` / `--filter nmap` | Filter by name |
| `--json` | Raw JSON output |
| `-o file` | Write output to file |

**Examples:** `tools`, `tools -f nmap`, `tools --json`, `tools -f sql --json -o sql_tools.json`

#### `python3 hexstrike.py status`

Check Pulse server health.

| Flag | Description | Default |
|------|-------------|---------|
| `--host 10.0.0.1` | Server address | `127.0.0.1` |
| `--port 9090` | Server port | `8888` |
| `--json` | Raw JSON output | off |
| `-o file` | Write output to file | |

**Example:** `status --host 192.168.1.10 --port 8080 --json -o health.json`

#### `python3 hexstrike.py validate`

Check which external tools are installed on PATH.

| Flag | Description |
|------|-------------|
| `-f nmap` / `--tool-filter nmap` | Check only matching tools |
| `-v` / `--verbose` | Show present tools too |
| `--json` | Raw JSON output |
| `-o file` | Write output to file |

**Examples:** `validate`, `validate -v`, `validate --json`, `validate -v --json -o report.json`

#### `python3 hexstrike.py mcp`

Stdio bridge for Claude Desktop (internal, used via `./hexstrike-pulse --bridge`).

| Flag | Description | Default |
|------|-------------|---------|
| `--server URL` | Pulse server URL | `http://127.0.0.1:8888` |
| `--timeout 600` | Request timeout (s) | `300` |
| `--debug` | Debug logging | off |
| `--compact` | Compact mode for small LLMs | off |
| `--profile web` | Tool profiles to load | `[]` |
| `--auth-token tok` | Bearer token | |
| `--disable-ssl-verify` | Disable SSL verification | off |

#### `python3 hexstrike.py ctf`

CTF challenge analysis workflow.

| Flag | Description | Default |
|------|-------------|---------|
| `--category pwn` | Category (web/pwn/crypto/forensics/re/misc) | `web` |
| `--name MyChallenge` | Challenge name | auto |
| `--description "..."` | Challenge description | auto |
| `--difficulty hard` | Difficulty (easy/medium/hard/insane) | `medium` |
| `--points 500` | Points | `0` |
| `--target 10.0.0.1` | Target IP/host | |
| `--json` | Raw JSON output | off |
| `-o file` | Write output to file | |

**Examples:** `ctf --category pwn --difficulty hard`, `ctf --category web --target http://10.0.0.1 --json`

---

## Example output

```
target:     scanme.nmap.org
intensity:  medium
surface:    2 open ports (22/ssh, 80/http) · risk: medium
findings:   [MEDIUM] missing-header · [INFO] python detected
plan:       8 steps · 15m est · 74% success probability
```

---

## Architecture

Pulse is organized into 6 modules under `pulse/`:

| Module | Role |
|--------|------|
| `pulse/interface/` | MCP server setup, tool binding, typed tool docs |
| `pulse/tools/` | 163 tool handlers, CTF engine, null context, tool registry |
| `pulse/intelligence/` | Decision engine, parameter optimizer, error correlation |
| `pulse/infrastructure/` | Cache, metrics, storage, logging, config |
| `pulse/server/` | HTTP server, CLI, MCP entry point, stdio bridge |
| `pulse/workflows/` | CTF challenge workflow, exploit rules |

---

## Legal

| Allowed | Not allowed |
|---------|-------------|
| Authorized penetration testing with written permission | Unauthorized testing of any system |
| Bug bounty programs within scope | Malicious or illegal activities |
| CTF competitions and labs | Unauthorized data access |
| Security research on owned systems | |

This software is intended solely for authorized security testing, research, and educational purposes.
