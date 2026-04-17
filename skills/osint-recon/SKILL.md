---
name: osint-recon
description: Open-source intelligence workflows for HexStrike OSINT tools. Use when the task is public-source investigation around domains, usernames, exposure, or internet-facing metadata rather than direct exploitation.
---

# OSINT Recon

## When to use

Use this skill when the user needs public-source collection, not active exploitation.

## Working Style

1. keep the question specific: username, org footprint, domain exposure, or robots policy
2. prefer public collection before active probing
3. hand off confirmed domains or hosts to `subdomain-enum`, `web-recon`, or `nmap-recon`

Preferred entrypoint:

```python
sherlock(username="targetuser")
```

See `REFERENCE.md` for common OSINT tool calls.
