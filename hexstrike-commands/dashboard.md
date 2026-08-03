---
description: /dashboard — Open the Pulse live dashboard
---

# /dashboard — Pulse Dashboard

Call the `pulse_dashboard()` tool via Pulse MCP.

## Usage

```
/dashboard
```

## Prompt

Call `pulse_dashboard()` to open the Pulse Prefab UI dashboard.

The dashboard is organized in 3 zones: Overview & Workflow (default view: scope, surface, plan, active tools, trends, cache, missing tools, rate limit), Findings (severity breakdown + details), and History (scans, sessions, errors, tool performance, confirmations).

Do not call any other tools — this single tool renders the full interface.
