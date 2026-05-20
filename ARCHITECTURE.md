# Architecture

This document describes how HexStrike AI-PULSE works at a high level — enough to understand the moving parts without internal jargon.

## Overview

```
You (CLI / AI Agent)
        │
        ▼
  ┌─────────────┐
  │  Entry Point │── CLI, HTTP server, or MCP bridge
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │    Engine   │── chooses tools, plans attacks, adapts parameters
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Executor   │── runs the actual security tool (nmap, nuclei, etc.)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Toolchain  │── 150+ installed binaries on your system
  └─────────────┘
```

Everything runs **locally** on your machine. No cloud, no accounts. Results stay on disk.

## Entry Points

Three ways to start a scan:

| Entry | When to use |
|-------|-------------|
| `hexstrike.py` | Direct CLI usage. Run one tool at a time, chain commands, script workflows. |
| `hexstrike_server.py` | HTTP server + JSON-RPC API. Needed for AI agent integration and the live dashboard. |
| `hexstrike-pulse` | Launcher for Claude Desktop. Wraps the MCP bridge with auto-venv and lock cleanup. |

All three load the same tool registry and execution engine.

## Data Pipeline

When you run a scan, data flows through four stages:

```md

  1. TOOL EXECUTION
     nmap, whatweb, nuclei, nikto run against the target
     stdout/stderr captured

  2. RESULT CACHE
     Every tool result is stored with tool name, target, output, and timestamp
     A cached target is never re-scanned — saves time on repeated work

  3. ANALYSIS
     Surface:  nmap output → open ports, services, technologies
     Findings: nuclei/nikto output → vulnerabilities sorted by severity
     Plan:     target profile + tool history → recommended attack chain

  4. PERSISTENCE
     Results are saved locally (JSON files on disk)
     Every scanned target keeps its history across sessions
```

This runs in **parallel where possible** — a full scan with all four tools completes in 2-5 minutes instead of the sequential 10+ minutes.

## Intelligence Layer

The engine doesn't just run tools blindly. It:

- **Selects tools** based on target type (IP vs domain vs URL) and scan intensity (quick / medium / full)
- **Adapts parameters** — when a WAF is detected, stealth mode activates automatically; when WordPress is found, relevant plugins get tested
- **Plans attacks** — generates a multi-step attack chain with success probability and time estimate per step
- **Tracks effectiveness** — learns which tools succeed or fail over time, prefers reliable tools

For example, a `full` scan on a web target runs:
1. **nmap** — port scanning + service detection
2. **whatweb** — technology fingerprinting
3. **nuclei** — vulnerability scanning (4000+ templates)
4. **nikto** — web server misconfiguration scanning
5. **gobuster** — directory enumeration

The planner then chains results from each tool into the next step, building an attack plan adapted to what was found.

## Live Dashboard

The HTTP server serves a real-time dashboard at `http://127.0.0.1:8888/dashboard` showing:

- **System resources** — CPU, RAM, disk usage with sparklines
- **Scope** — current active target, type, scan history
- **Surface** — open ports, services, risk level
- **Findings** — vulnerabilities detected, severity breakdown
- **Attack plan** — recommended next steps with probabilities
- **Tool health** — success rates, timeouts, slowest tools
- **Cache status** — hit rate, entry count, per-tool usage
- **Intelligence** — per-tool effectiveness scores (baseline vs live)

## Safety

Destructive tools require explicit confirmation before execution:

- `aireplay-ng` (WiFi deauth)
- `metasploit` (exploitation)
- `responder` (LLMNR poisoning)
- `mdk4` (WiFi stress testing)
- `mitm6` (IPv6 takeover)

Each tool runs as a subprocess with timeout protection. No tool runs longer than its configured timeout (default 5 minutes).

## What HexStrike Doesn't Do

- **No cloud** — zero data leaves your machine. No accounts, no API keys, no telemetry.
- **No user management** — single-user, local-only. No teams, no permissions, no audit log.
- **No GUI** — the dashboard is a web page, but all interaction happens through CLI or your AI agent.
- **No persistence** beyond local JSON files — results live in `data/` relative to where you started the server. Stop the server and restart elsewhere? New data directory.
