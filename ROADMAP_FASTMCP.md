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
Prompts, real-time Context streaming, and interactive Apps. The Flask layer is now
redundant and actively limits what the LLM can see and do.

This roadmap proposes a clean, non-breaking 4-phase migration.

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

### 5. `Resources` and `Prompts` — workflow intelligence (Phase 3+)

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

### 6. FastMCP Apps — interactive UI in the conversation (Phase 4)

```python
from fastmcp import FastMCP
from prefab_ui.components import Column, Heading, BarChart, Table

@mcp.tool(app=True)
def hexstrike_dashboard() -> PrefabApp:
    """Live HexStrike dashboard — server health, tools, running scans."""
    ...
```

MCP Apps render interactive UIs directly inside the MCP client conversation
(Claude Desktop, VS Code Copilot, etc.) — no browser, no separate frontend.

---

## Current state of `refactor/fastmcp-modernization`

### Done (22 commits ahead of beta/1.0.13)

| What | Files | Status |
|------|-------|--------|
| Simplify MCP startup for FastMCP 3.1+ | `mcp_core/mcp_entry.py`, `server_setup.py` | ✅ |
| `mcp.run(show_banner=False, log_level="WARNING")` | `mcp_core/mcp_entry.py` | ✅ |
| `BM25SearchTransform` for tool discovery | `mcp_core/server_setup.py` | ✅ |
| `SkillsDirectoryProvider` for skills/ | `mcp_core/server_setup.py` | ✅ |
| `ctx: Context` migration — 62 tools across all non-wifi modules | `mcp_tools/` | ✅ |
| WiFi pentest docstrings — workflow context | all 12 wifi tools | ✅ |
| Stub `FileUploadTestingFramework` | `server_core/workflows/bugbounty/testing.py` | ✅ |

### In progress

| What | Status |
|------|--------|
| `ctx: Context` + `run_in_executor` — wifi_pentest (12 tools) | 🔄 needs redo after rebase |
| Unit tests — wifi_pentest mcp_tools | 🔄 in progress |

### Known divergence from upstream

Upstream `beta/1.0.13` added `run_in_executor` to all tools (correct async pattern).
Our branch has `ctx: Context` but wifi tools lost it during rebase conflict resolution.
Both are needed — the correct pattern is:

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
    → HTTP → Flask → server_api/wifi_pentest/airmon_ng.py → subprocess

After (Phase 1):
  mcp_tools/wifi_pentest/airmon_ng.py
    → FastMCP Client → server_api/wifi_pentest/airmon_ng.py → subprocess (direct)
```

**Why wifi_pentest first:**
- Cleanest module — added in beta/1.0.12, no V6 legacy
- All tools follow the same pattern
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

Resources and Prompts are introduced here to enable guided workflows:

```python
@mcp.prompt()
async def bug_bounty_workflow(target: str, scope: str) -> list:
    """Full bug bounty recon → vuln hunt → report workflow."""
    ...

@mcp.prompt()
async def ctf_workflow(challenge_type: str) -> list:
    """CTF challenge solving workflow — binary/web/crypto/forensics."""
    ...
```

---

## Phase 4 — FastMCP Apps: interactive dashboard in the conversation

**Goal:** Replace the `ui/` React/Vite/Flask dashboard with a FastMCP App
that renders directly inside the MCP client conversation — no browser, no
separate frontend, no `npm build`.

### Why this is better than the current `ui/` approach

| Current `ui/` (React + Flask) | FastMCP App (Phase 4) |
|-------------------------------|----------------------|
| Separate React SPA — `npm build` + static server | `pip install "fastmcp[apps]"` |
| Calls Flask via HTTP | Tool returns UI directly |
| User opens a browser tab | UI appears in the conversation |
| LLM cannot see or interact with the UI | LLM sees data AND renders UI |
| Two separate apps to maintain | One unified MCP server |

### What the HexStrike dashboard App will do

```python
from fastmcp import FastMCP
from prefab_ui.components import (
    Column, Row, Heading, Badge, BarChart, Table,
    Button, ProgressBar, ChartSeries
)
from prefab_ui.app import PrefabApp

@mcp.tool(app=True)
async def hexstrike_dashboard(ctx: Context) -> PrefabApp:
    """
    Live HexStrike operations dashboard.

    Shows server health, tool availability, running scans,
    and provides controls to launch/kill operations.
    """
    health = await get_health_async()
    running = await get_running_scans_async()
    resources = await get_resource_usage_async()

    with Column(gap=4) as view:

        # ── Header ──────────────────────────────────────────────────────
        Row(
            Heading("🔥 HexStrike Dashboard"),
            Badge(health["status"], color="green" if health["status"] == "healthy" else "red"),
        )

        # ── System resources ────────────────────────────────────────────
        Heading("System", level=2)
        Row(
            ProgressBar(label="CPU",    value=resources["cpu_percent"],    max=100),
            ProgressBar(label="Memory", value=resources["memory_percent"], max=100),
            ProgressBar(label="Disk",   value=resources["disk_percent"],   max=100),
        )

        # ── Tool availability by category ───────────────────────────────
        Heading("Tools", level=2)
        BarChart(
            data=[
                {"category": k, "available": v["available"], "total": v["total"]}
                for k, v in health["category_stats"].items()
            ],
            series=[
                ChartSeries(data_key="available", label="Available"),
                ChartSeries(data_key="total",     label="Total"),
            ],
            x_axis="category",
        )

        # ── Running scans ───────────────────────────────────────────────
        if running:
            Heading("Running Scans", level=2)
            Table(
                columns=["tool", "target", "pid", "started"],
                rows=running,
                actions=[
                    {"label": "Kill", "tool": "kill_scan", "param": "pid"},
                ]
            )

        # ── Quick launch ────────────────────────────────────────────────
        Heading("Quick Launch", level=2)
        Row(
            Button(label="Health Check",  tool="check_health"),
            Button(label="Refresh Tools", tool="refresh_tool_availability"),
            Button(label="Clear Cache",   tool="clear_cache"),
        )

    return PrefabApp(view=view)
```

### What the LLM gains

Because the dashboard is a tool result, the LLM can:
- Read the health data and proactively suggest fixes
- Notice a missing tool and offer to install it
- See a running scan and ask if you want to kill it
- Correlate resource usage with scan load

This is not possible with the current React/Flask approach where the UI
and the LLM are completely separate.

### Dependencies for Phase 4

```
pip install "fastmcp[apps]" prefab-ui
```

Note: `prefab-ui` is in early development — pin to a specific version.

---

## PR checklist (before opening)

- [ ] `ctx: Context` + `run_in_executor` on all migrated tools (including wifi)
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
- FastMCP Apps: https://gofastmcp.com/apps/overview
- Prefab UI: https://prefab.prefect.io
