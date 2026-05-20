# HexStrike AI-PULSE

AI-powered security orchestration engine with a live dashboard. Connect your AI agent, describe your objective, and let it orchestrate 150+ security tools — from recon to exploitation — all visible in real time.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![MCP](https://img.shields.io/badge/MCP-Compatible-purple)]()
[![License](https://img.shields.io/badge/License-AGPLv3-green)]()

---

## Quick Install

```bash
git clone https://github.com/VhaTer/hexstrike-ai-community-edition.git
cd hexstrike-ai-community-edition
python3 -m venv hexstrike-env
source hexstrike-env/bin/activate
pip install -r requirements.txt
```

**Prerequisites:** Linux (Kali/Debian/Ubuntu), Python 3.11+, and security tools installed on your system (`nmap`, `whatweb`, `nuclei`, `nikto`, `gobuster`, `sqlmap`). Run `python3 hexstrike.py validate` to check what's available.

---

## What you can do

### Orchestrate — let AI run the operation

Start the server, connect your AI agent, and describe your objective:

```bash
python3 hexstrike_server.py
```

Then in Claude, OpenCode, or any MCP client:

```
> I'm a security researcher. My company owns scanme.nmap.org.
  Show me the attack surface and any critical vulnerabilities.
```

The agent runs nmap, whatweb, nuclei, and nikto in sequence, analyzes results, and returns a structured report — ports, services, technologies, vulnerabilities, and recommended next steps. All cached so re-asking is instant.

### Monitor — live Pulse dashboard

Open `http://127.0.0.1:8888/dashboard` while scans run. See:

- Open ports and risk level as they're discovered
- Technologies detected per target
- Vulnerabilities found, sorted by severity
- Tool performance and success rates
- Cache hit rate and system resources

Every scan populates the dashboard in real time. No refresh needed.

### Execute — CLI when you need direct control

```bash
# Port scan
python3 hexstrike.py scan nmap scanme.nmap.org

# Technology detection
python3 hexstrike.py scan whatweb http://example.com

# Vulnerability scan
python3 hexstrike.py scan nuclei http://example.com -p severity=medium
```

Full tool list: `python3 hexstrike.py tools`

---

## Understanding results

A scan returns four sections:

| Section | What it contains | Source |
|---------|------------------|--------|
| **Tools** | Status per tool (completed / cached / failed / skipped) | Execution |
| **Surface** | Open ports, services, technologies, risk level | nmap + whatweb |
| **Findings** | Vulnerabilities sorted by severity | nuclei + nikto |
| **Plan** | Attack chain with success probability and time estimates | AI engine |

### Example

```
target:     scanme.nmap.org
intensity:  medium
tools:      nmap: completed · whatweb: completed · nuclei: completed · nikto: cached

surface:    2 open ports (22/ssh, 80/http) · risk: medium
findings:   [MEDIUM] missing-header · [INFO] python detected
plan:       8 steps · 15m est · 74% success probability
```

---

## Connect your AI agent

**Claude Desktop**
```json
{
  "mcpServers": {
    "hexstrike-pulse": {
      "type": "http",
      "url": "http://127.0.0.1:8888/mcp"
    }
  }
}
```

WSL variant (Windows):
```json
{
  "mcpServers": {
    "hexstrike-pulse": {
      "command": "wsl.exe",
      "args": ["bash", "-ic", "cd /path/to/hexstrike && ./hexstrike-pulse"],
      "type": "stdio"
    }
  }
}
```

**OpenCode**
```json
{
  "mcp": {
    "hexstrike-pulse": {
      "type": "remote",
      "url": "http://127.0.0.1:8888/mcp",
      "enabled": true
    }
  }
}
```

**VS Code / Cursor / Roo Code**
```json
{
  "servers": {
    "hexstrike-pulse": {
      "type": "http",
      "url": "http://127.0.0.1:8888/mcp"
    }
  }
}
```

---

## Entry points

| Command | Use case |
|---------|----------|
| `hexstrike.py` | CLI — run tools, validate setup, list tools |
| `hexstrike_server.py` | HTTP server + dashboard + MCP endpoint |
| `hexstrike-pulse` | Launcher for Claude Desktop (auto-venv, lock cleanup) |
| `hexstrike_mcp.py` | MCP stdio bridge (debug) |

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a high-level overview of the data pipeline, intelligence layer, and safety model.

---

## Legal

| Allowed | Not allowed |
|---------|-------------|
| Authorized penetration testing with written permission | Unauthorized testing of any system |
| Bug bounty programs within scope | Malicious or illegal activities |
| CTF competitions and labs | Unauthorized data access |
| Security research on owned systems | |

This software is intended solely for authorized security testing, research, and educational purposes.
