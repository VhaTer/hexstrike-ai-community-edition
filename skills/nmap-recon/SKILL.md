---
name: nmap-recon
description: Network and service reconnaissance workflows for HexStrike tools. Use when you need host discovery, port discovery, service enumeration, or scripted Nmap follow-up.
---

# Nmap Recon

## When to use

Use this skill when the task is primarily network reconnaissance:

- identify live hosts on a subnet
- discover open TCP or UDP ports
- enumerate service banners and versions
- follow up with targeted NSE scripts

## Working Style

Start broad, then narrow:

1. use `arp_scan`, `rustscan`, or `masscan` to discover hosts or ports quickly
2. confirm and enrich with `nmap`
3. use `nmap_advanced` only once you know which services justify extra NSE traffic

Prefer the runtime entrypoint:

```python
nmap(target="10.10.10.10", ports="22,80,443", scan_type="-sCV")
```

For the exact parameter shapes and copy-pasteable calls, read `REFERENCE.md`.

## Notes

- keep scans scoped to authorised targets only
- start with lower-noise modes before aggressive timing or broad NSE runs
- if a user asks how to approach recon, explain the sequence first instead of jumping straight to a noisy full-port scan
