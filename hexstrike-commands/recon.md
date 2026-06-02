---
description: /recon — Consolidated surface analysis + findings + attack plan via Pulse
---

# /recon — Consolidated Reconnaissance

Call `get_surface()`, `get_findings()`, and `get_plan()` via Pulse MCP.

## Usage

```
/recon <target>
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| target | — (required) | IP, domain, or URL |

## Prompt

Target: $1

Run a consolidated analysis on $1 using Pulse MCP tools:

1. Call `get_surface(target="$1")` — open ports, services, technologies.
2. Call `get_findings(target="$1")` — vulnerability findings sorted by severity.
3. Call `get_plan(target="$1", objective="comprehensive")` — suggested attack chain with probability estimates.
4. Present a structured summary: surface overview, critical findings, recommended next tool.
