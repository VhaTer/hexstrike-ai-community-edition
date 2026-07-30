# HexStrike AI-PULSE

AI-powered security orchestration engine — 150+ tools, live Prefab dashboards, any MCP client.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![MCP](https://img.shields.io/badge/MCP-Streamable%20HTTP-purple)]()
[![Coverage](https://img.shields.io/badge/Coverage-99%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-AGPLv3-green)]()

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

Open `http://127.0.0.1:8888/dashboard` while scans run. 10 panels across 3 tabs:

| Tab | Panels |
|-----|--------|
| **Overview** | Header, Scope, Surface, Findings, System Trends, Cache Status, Intelligence |
| **Workflow** | Plan IDE, Active Tools, Async Scans, Missing Tools, Rate Limit |
| **History** | History, Sessions, Errors & Failures, Tool Performance, Confirmations |

### Execute

```bash
python3 hexstrike.py scan nmap scanme.nmap.org
python3 hexstrike.py scan whatweb http://example.com
python3 hexstrike.py scan nuclei http://example.com -p severity=critical
python3 hexstrike.py tools
```

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
| `pulse/tools/` | 150+ tool handlers, CTF engine, null context, tool registry |
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
