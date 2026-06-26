# HexStrike AI-PULSE

AI-powered security orchestration engine with live Prefab dashboards. Connect Claude Desktop, OpenCode, Continue.dev, or any MCP client — describe your objective, and let AI orchestrate 150+ security tools in real time.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![MCP](https://img.shields.io/badge/MCP-Streamable%20HTTP-purple)]()
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

**Prerequisites:** Linux (Kali/Debian/Ubuntu), Python 3.11+, and security tools installed (`nmap`, `whatweb`, `nuclei`, `nikto`, `gobuster`, `sqlmap`, …). Run `python3 hexstrike.py validate` to check availability.

---

## Launch

Start the HTTP server — one terminal, stays up:

```bash
./hexstrike-pulse
```

| Command | What it does |
|---------|-------------|
| `./hexstrike-pulse` | Start HTTP server in background |
| `./hexstrike-pulse --foreground` | Start with visible logs |
| `./hexstrike-pulse stop` | Stop the server |
| `./hexstrike-pulse status` | Check if running |
| `./hexstrike-pulse --bridge` | Stdio→HTTP bridge (for Claude Desktop) |

All clients connect to the same server — no conflicts, no duplicate lock files.

### Entry points at a glance

| Entry point | When to use |
|---|---|
| `./hexstrike-pulse` | **Default** — HTTP server in background, one command |
| `python3 hexstrike.py serve` | CLI alternative, foreground with logs |
| `python3 hexstrike_server.py` | Dev/debug — Starlette HTTP + dashboard |
| `python3 hexstrike_mcp.py` | Internal — stdio mode for Claude Desktop |
| `python3 hexstrike.py scan <tool> <target>` | Direct tool execution, no server needed |

Use `./hexstrike-pulse` unless you have a specific reason for another.

---

## Connect your AI agent

### Claude Desktop (WSL/Windows)

```json
{
  "mcpServers": {
    "hexstrike-pulse": {
      "command": "wsl.exe",
      "args": ["-d", "kali-linux", "/path/to/hexstrike-ai-community-edition/hexstrike-pulse", "--bridge"]
    }
  }
}
```

The `--bridge` flag runs `pulse-bridge.py` — a lightweight stdio→HTTP proxy with zero heavy imports. It connects to the already-running HTTP server instantly (~0.3s init), bypassing the 5s Python import overhead.

Replace `kali-linux` with your WSL distro name and `/path/to` with the output of `pwd`.

### OpenCode

```json
{
  "mcp": {
    "hexstrike-pulse": {
      "type": "remote",
      "url": "http://localhost:8888/mcp",
      "enabled": true
    }
  }
}
```

### Continue.dev / Cline / any MCP client

```json
{
  "servers": {
    "hexstrike-pulse": {
      "type": "streamable-http",
      "url": "http://localhost:8888/mcp"
    }
  }
}
```

All clients share the same HTTP server — openCode via remote URL, Claude Desktop via stdio bridge, Continue/Cline via direct HTTP. Zero conflicts.

---

## What you can do

### Orchestrate — let AI run the operation

In your AI agent, describe your objective:

```
> I'm a security researcher. My company owns scanme.nmap.org.
  Show me the attack surface and any critical vulnerabilities.
```

The agent runs nmap, whatweb, nuclei, and nikto in sequence, analyzes results, and returns ports, services, technologies, vulnerabilities, and recommended next steps. All cached.

### Monitor — live dashboard

Open `http://127.0.0.1:8888/dashboard` while scans run. See open ports, risk levels, tool performance, cache rates, and system resources in real time.

### Execute — CLI when you need direct control

```bash
python3 hexstrike.py scan nmap scanme.nmap.org
python3 hexstrike.py scan whatweb http://example.com
python3 hexstrike.py scan nuclei http://example.com -p severity=medium
```

Full tool list: `python3 hexstrike.py tools`

---

## Understanding results

| Section | Content | Source |
|---------|---------|--------|
| **Tools** | Status per tool (completed / cached / failed / skipped) | Execution |
| **Surface** | Open ports, services, technologies, risk level | nmap + whatweb |
| **Findings** | Vulnerabilities sorted by severity | nuclei + nikto |
| **Plan** | Attack chain with probability and time estimates | AI engine |

Example output:

```
target:     scanme.nmap.org
intensity:  medium
surface:    2 open ports (22/ssh, 80/http) · risk: medium
findings:   [MEDIUM] missing-header · [INFO] python detected
plan:       8 steps · 15m est · 74% success probability
```

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the data pipeline, intelligence layer, and safety model.

---

## Session Reports & Development Notes

Detailed session reports and architectural notes are stored in `Projects_reports_docs/`:
- CTF analyses (local, not in git)
- Session briefs and design decisions

See `AGENTS.md` for development discipline rules and orchestration patterns.

---

## Legal

| Allowed | Not allowed |
|---------|-------------|
| Authorized penetration testing with written permission | Unauthorized testing of any system |
| Bug bounty programs within scope | Malicious or illegal activities |
| CTF competitions and labs | Unauthorized data access |
| Security research on owned systems | |

This software is intended solely for authorized security testing, research, and educational purposes.
