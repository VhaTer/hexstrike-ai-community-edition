# HexStrike AI-PULSE — Technology Stack Audit

> Audited 2026-07-31. Method: AST import scan over `pulse/`, cross-referenced with
> `requirements.txt` and `pipdeptree`. Numbers verified manually by review.

## Architecture in one line

**Python orchestrates, external binaries execute.** The Python layer (`pulse/`)
is a thin orchestration core built on FastMCP 3; the 163 tools are external
binaries invoked via subprocess through `run_security_tool` → `*_direct.py`.

## Layer 1 — Core Python libraries (directly imported by `pulse/`)

| Library | Version | Imports | Role |
|---|---|---|---|
| `fastmcp[apps]` | 3.2.4 | 17 | MCP server framework + Prefab UI integration |
| `requests` | 2.32.x | 13 | HTTP client for API-style tools |
| `psutil` | 5.9.x | 8 | System stats (processes, CPU, memory) |
| `prefab_ui` | 0.14.x | 4 | Dashboard UI components (pinned for API stability) |
| `beartype` | 0.18.x | 4 | Runtime type checking (@beartype) |

Occasional direct imports (specialized paths):

| Library | Imports | Where |
|---|---|---|
| `angr` | 2 | Binary analysis (CTF pwn) |
| `pymysql` | 1 | Database interactions |
| `playwright` | 1 | Browser automation |
| `starlette` | 1 | Via fastmcp (transitive, re-exported) |

## Layer 2 — Declared in `requirements.txt` but never imported (12)

These are **invoked as external binaries** (subprocess), not used as Python
libraries:

`bbot`, `bcrypt`, `chardet`, `checkov`, `dirsearch`, `httpx`, `one_gadget`,
`pwntools`, `ropgadget`, `social_analyzer`, `uro`, `volatility3`

Notes:

- `bcrypt` (0 imports) is pinned only for passlib/pwntools transitive
  compatibility — declared for dependency reasons, not direct use.
- `httpx` (0 direct imports in `pulse/`) is already pulled transitively by
  `fastmcp` (`mcp` → `httpx >=0.27.1`) — the explicit requirement is redundant
  but harmless.
- Counting these 12 in `requirements.txt` as "Python dependencies" overstates
  the runtime surface: the real Python dependency footprint is 5 core libs.

## Layer 3 — External security tools (163, not pip)

Registered in `tool_registry.py` (163 entries, 20 categories), invoked as
system binaries:

| # | Category | # | Category |
|---|---|---|---|
| 20 | web_recon | 12 | active_directory |
| 20 | forensics | 8 | essential |
| 16 | wifi_pentest | 6 | exploitation |
| 15 | osint | 6 | api |
| 14 | network_recon | 5 | brute_force |
| 13 | binary | 4 | intelligence |
| 13 | cloud | 4 | web_vuln |

Smaller categories: database (2), vulnerability_intelligence, lateral_movement,
ops, fingerprint, primitive (1 each).

Examples: `nmap`, `masscan`, `gobuster`, `ffuf`, `nuclei`, `nikto`, `sqlmap`,
`hydra`, `john`, `ghidra`, `radare2`, `prowler`, `trivy`, `sherlock`.

## Environment snapshot (dev venv)

- `pip list`: 339 packages installed
- `pipdeptree` top-level: 57 packages — only **5 are imported by `pulse/`**
  (Layer 1). The other 52 are: dev/test tooling (`pytest*`, `ruff`,
  `pipdeptree`), Layer 2 binaries, and legacy V6 residue.
- Key transitive chain: `fastmcp` → `mcp` (1.26.0) → `httpx`, `anyio`,
  `pydantic`; → `prefab_ui` (dashboard), → `starlette` (HTTP transport).

## Code layout (pulse/)

```
pulse/
├── interface/     # @app.tool() definitions, server_setup.py (orchestrator)
├── tools/         # *_direct.py — subprocess wrappers for the 163 binaries
├── intelligence/  # decision engine, attack chains, exploit rules, CVE intel
├── infrastructure/
│   ├── storage/   # caches, target/session/tool_stats/wordlist stores
│   ├── execution/ # command executors, process pools, recovery
│   ├── telemetry/ # metrics, performance monitoring, dashboards
│   └── ...        # config, singletons, logging, middleware, error handling
├── server/        # entry points: cli.py, mcp_server.py, web_server.py, bridge.py
└── workflows/     # CTF workflows (challenge, tool_manager, coordinator...)
```

Root-level legacy modules still imported: `config.py`, `tool_registry.py`
(pre-package restructure, V6-era).

## Dependency hygiene summary

- True Python deps: **5** (fastmcp, requests, psutil, prefab_ui, beartype)
- pip-installed but used as binaries: **12** (Layer 2)
- pip-installed, unused in `pulse/`: ~40 of the 57 top-level (dev tooling +
  V6 residue)
- External binaries not pip at all: **163 tools** (Layer 3)
