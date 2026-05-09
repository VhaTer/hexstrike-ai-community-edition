# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

HexStrike Pulse is a cybersecurity assessment tool. If you discover a security vulnerability:

1. **Do not** open a public GitHub issue.
2. Email the maintainers at `security@hexstrike.ai` (or open a private advisory via GitHub).
3. Include a clear description, reproduction steps, and affected version.
4. You should receive a response within 48 hours.

## Security Features

### Destructive Tool Confirmation

Five tools require explicit user confirmation before execution:

- `aireplay_ng` — Wi-Fi deauthentication
- `mdk4` — Wi-Fi flooding
- `metasploit` — Exploit delivery
- `responder` — LLMNR/NBT-NS poisoning
- `mitm6` — IPv6 DNS hijacking

These tools are tagged with `requires_confirmation=True` in the tool registry and will prompt before execution.

### Rate Limiting

The `RateLimitDetector` module (`server_core/rate_limit_detector.py`) automatically detects HTTP 429 responses and adjusts timing profiles to avoid overwhelming targets.

### Scan Cache

In-memory scan cache (`_ScanCache`, max_size=500, default_ttl=1800s) prevents redundant scanning. Cache keys are param-hashed (MD5). Cache is per-session and cleared on server restart.

### Input Validation

- Tool parameter validation via configured JSON schemas
- CVE ID format validation (`CVE-YYYY-NNNNN`)
- URL sanitization in fuzzing tools (prevents `/FUZZ/FUZZ` duplication)

## Deployment Security

- The HTTP/SSE server (`hexstrike_server.py`) listens on `127.0.0.1:8888` by default.
- Override via `HEXSTRIKE_HOST` / `HEXSTRIKE_PORT` environment variables.
- Do not expose the server to untrusted networks. It has no authentication layer.
- The stdio MCP interface (`hexstrike_mcp.py`) is designed for local use only.

## Responsible Use

HexStrike Pulse is intended for authorized security testing only. Users are responsible for:

- Obtaining written permission before testing any target
- Complying with all applicable laws and regulations
- Using destructive tools only against explicitly authorized targets
