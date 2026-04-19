---
name: web-recon
description: Web application reconnaissance via endpoint mapping, content discovery, and technology fingerprinting. Use to build complete request surface before vulnerability scanning.
---

# Web Recon

## When to use

Use this skill in **Phase 2: Enumeration** when mapping HTTP/HTTPS application surface:

- Fingerprint web technologies, WAF, and server configuration
- Enumerate hidden content (directories, files, backup copies)
- Extract parameters, forms, and endpoints via crawling/spidering
- Identify virtual hosts and subdomain patterns
- Collect historical endpoints via archived sources

## Working Style

**Controlled expansion prevents WAF triggers and noise:**

1. **Fingerprint** — `httpx -p -t` or `whatweb` → detect tech, WAF, SSL config
2. **Check WAF** — `wafw00f` automatically triggers adaptive parameter optimization (stealth mode)
3. **Content Discovery** — `ffuf` or `feroxbuster` with detected tech wordlists; avoid aggressive `-t` unless authorized
4. **Crawling** — `katana` (modern) or `waybackurls` (historical); then `arjun`/`paramspider` for params
5. **Handoff** → `web-vuln` with refined URL list

**Entry point:**

```python
httpx(target="app.example.com", ports="80,443", probe=True, tech_detect=True)
```

## Notes

- **WAF Avoidance:** Always detect WAF first; HexStrike auto-applies stealth mode when detected
- **Effectiveness:** httpx (0.95), ffuf (0.90), katana (0.92); dalfox crawl has 15% FP on XSS
- **Wordlists:** Adapt to detected CMS (WordPress → wp-content, Drupal → sites/default)
- **False Positives:** Collect 404 baseline before brute-forcing to filter noise
- **Avoid:** Brute-forcing before WAF check; aggressive timing on unverified targets
