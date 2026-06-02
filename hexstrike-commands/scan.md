---
description: /scan — Full reconnaissance scan on a target via Pulse
---

# /scan — Pulse Reconnaissance

Call the `scan()` tool via Pulse MCP.

## Usage

```
/scan <target> [intensity]
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| target | — (required) | IP, domain, or URL |
| intensity | quick | quick (nmap+whatweb) / medium (+nuclei+nikto) / full (+gobuster) |

## Prompt

Target: $1
Intensity: ${2:-quick}

1. Use `search_tools("scan")` to find the scan tool.
2. Call `scan(target="$1", intensity="${2:-quick}", objective="comprehensive")`.
3. Present the results: open ports, technologies, vulnerabilities, and next steps.
4. If findings contain credentials (password=), suggest `authenticate()` via http_request.
5. For long-running scans (>30s), use `scan_background()` instead.
