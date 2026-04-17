---
name: web-recon
description: Web application reconnaissance workflows for HexStrike tools. Use when you need HTTP probing, content discovery, crawling, parameter discovery, vhost enumeration, or lightweight fingerprinting.
---

# Web Recon

## When to use

Use this skill when a target is primarily HTTP or HTTPS and the goal is to map surface area before exploitation.

## Working Style

1. fingerprint first with `httpx`, `whatweb`, `wafw00f`, or `testssl`
2. choose content discovery with `ffuf`, `feroxbuster`, `gobuster`, or `dirsearch`
3. crawl and expand with `katana`, `gau`, `waybackurls`, `arjun`, `paramspider`, or `x8`
4. hand off interesting endpoints to `web-vuln`

Preferred entrypoint:

```python
httpx(target="app.example.com", probe=True, tech_detect=True)
```

For concrete calls and common parameters, read `REFERENCE.md`.

## Notes

- do not brute-force blindly before checking whether a WAF is present
- adapt extensions and wordlists to detected tech
- `anew` and `uro` are cleanup helpers and should be used to reduce noise, not as primary recon steps
