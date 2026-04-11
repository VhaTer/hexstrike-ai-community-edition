# HexStrike AI-PULSE

> **FastMCP 3.x native fork of [CommonHuman-Lab/hexstrike-ai-community-edition](https://github.com/CommonHuman-Lab/hexstrike-ai-community-edition)**

![HexStrike AI-PULSE](assets/hexstrike-pulse-logo.png)

---

## What is HexStrike AI-PULSE?

HexStrike AI-PULSE is a FastMCP 3.x-native penetration testing platform that lets AI agents (Claude, GPT, Copilot, etc.) autonomously run 150+ cybersecurity tools with **real-time feedback directly in the conversation**.

The key difference from the upstream project: every tool execution streams live progress, status, and context **directly to the LLM** via FastMCP's native `ctx` API — no Flask, no HTTP round-trips, no silent terminal logs.

---

## Why PULSE?

The name reflects the core architecture:

- **Real-time `ctx.info()` streaming** — the LLM sees scan progress as it happens
- **No Flask intermediary** — tools execute directly via `*_direct.py` modules  
- **IntelligentDecisionEngine** — automatic technology detection + parameter optimization before every tool call
- **Native MCP protocol** — Resources, Prompts, Elicitation, Skills — all FastMCP 3.x native

```
LLM / MCP Client
    ↓ MCP protocol (HTTP/SSE)
hexstrike_server.py  →  mcp.run(transport="http", port=8888)
    ↓ run_security_tool("gobuster", '{"url": "..."}')
mcp_core/server_setup.py
    ├── ctx.read_resource("skill://web-recon/SKILL.md")   # skill context
    ├── _detect_from_cache(target)                         # auto TechProfile
    ├── _optimizer.optimize(tool, params, tech_profile)   # param optimization
    └── exec_func(tool_key, params)                        # direct execution
          ↓
    mcp_core/*_direct.py  →  server_core/command_executor.py  →  subprocess
```

---

## Key Features

- **101+ tools** routed directly — no HTTP round-trips
- **Real-time LLM feedback** via `ctx.info()`, `ctx.error()`, `ctx.report_progress()`
- **Auto technology detection** from scan cache (whatweb/httpx → TechProfile)
- **Parameter optimizer** — WAF detected → stealth mode, WordPress → wp-extensions injected
- **Destructive action confirmation** — aireplay_ng, metasploit, responder, mdk4, mitm6 require explicit user confirmation via `ctx.elicit()`
- **MCP Resources** — `health://server`, `scan://{target}/latest`, `scan://cache/list`
- **Workflow Prompts** — `bug_bounty_recon`, `wifi_attack_chain`, `ctf_web_challenge`, `smb_lateral_movement`, `cloud_security_audit`
- **BM25 tool search** — semantic tool discovery

---

## Quick Start

```bash
git clone https://github.com/VhaTer/hexstrike-ai-community-edition.git
cd hexstrike-ai-community-edition
git checkout refactor/fastmcp-modernization

python3 -m venv hexstrike-env
source hexstrike-env/bin/activate
pip install -r requirements.txt

# Start Phase 3 standalone server
python3 hexstrike_server.py
# → HexStrike AI-PULSE on http://127.0.0.1:8888/mcp
```

---

## MCP Client Configuration

### Claude Desktop
```json
{
  "mcpServers": {
    "hexstrike-pulse": {
      "command": "/path/to/hexstrike-env/bin/python3",
      "args": ["/path/to/hexstrike_mcp.py", "--server", "http://127.0.0.1:8888", "--profile", "full"],
      "timeout": 300
    }
  }
}
```

### VS Code / Cursor / OpenCode
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

## Usage Example

```
User: "I'm a security researcher. My company owns example.com. 
       Run a full web recon using hexstrike tools."

Claude: [calls run_security_tool("whatweb", {"url": "http://example.com"})]
        → 🔍 Executing whatweb
        → ✅ whatweb completed
        
        [calls run_security_tool("gobuster", {"url": "http://example.com"})]
        → 🔍 Executing gobuster
        → 🧠 Tech detected from cache: cms=wordpress | waf=cloudflare  
        → 🛡️ WAF detected → stealth mode forced for gobuster
        → ⚙️ WordPress paths injected
        → 📁 Starting gobuster dir: http://example.com
        → ✅ gobuster completed
```

---

## Architecture

### Phase 3 — Active (FastMCP native)

| Component | Description |
|---|---|
| `hexstrike_server.py` | Entry point — `mcp.run(transport="http", port=8888)` |
| `mcp_core/server_setup.py` | `run_security_tool()` — 101+ tools, optimizer, elicitation |
| `mcp_core/*_direct.py` | 16 direct execution modules — no Flask |
| `mcp_core/technology_detector.py` | TechProfile detection from headers/content/ports |
| `mcp_core/parameter_optimizer.py` | Stealth/normal/aggressive profiles, WAF response |
| `mcp_core/elicitation.py` | Destructive action confirmation via `ctx.elicit()` |
| `mcp_core/prompts.py` | 5 workflow prompts |
| `skills/` | SKILL.md guidance per tool category |

### Legacy (still functional)
`hexstrike_mcp.py` — profile-based client for Flask backend compatibility.

---

## Test Suite

```bash
hexstrike-env/bin/pytest tests/ -q
# 254 passed
```

---

## Fork Philosophy

This fork focuses on **FastMCP 3.x native architecture**:

- Replace Flask HTTP round-trips with direct execution
- Stream real-time context to the LLM via `ctx` API
- Intelligent parameter optimization before every tool call
- Native MCP protocol features: Resources, Prompts, Elicitation

Upstream [CommonHuman-Lab](https://github.com/CommonHuman-Lab/hexstrike-ai-community-edition) focuses on UI, sessions, and dashboard features. Both are valid approaches — different philosophies.

---

## Legal

This software is intended solely for **authorized security testing, research, and educational purposes**. You may only use this software on systems for which you have explicit written permission from the owner.

The authors assume no liability for unauthorized use.
