---
name: subdomain-enum
description: Passive and active subdomain discovery via DNS, certificates, and source harvesting. Use to expand internet-facing attack surface before web recon.
---

# Subdomain Enum

## When to use

Use this skill in **Phase 1: Reconnaissance** for external asset discovery:

- Passive subdomain collection (DNS, certificates, historical sources)
- Zone transfer attempts and DNS bruteforce (only on authorized targets)
- Subdomain takeover validation
- Virtual host enumeration via DNS/HTTP

## Working Style

**Passive-first avoids detection; active only when authorized:**

1. **Passive** — `subfinder` (50+ sources) or `amass` (passive mode); fast, undetectable
2. **Historical** — `theharvester` or `dnsenum` (cert history, archived DNS)
3. **Active DNS** — `fierce` or `dnsbrute` (zone transfer, dict attack) only if scope includes active DNS
4. **Validation** — Probe live subdomains with `httpx`, then hand off to `web-recon`
5. **Cleanup** — Deduplicate and normalize before downstream tools

**Entry point:**

```python
subfinder(domain="example.com", passive=True)
```

## Notes

- **Stealth:** Passive sources are invisible; active DNS enumeration is immediately logged
- **Effectiveness:** subfinder (0.92), amass passive (0.90), fierce (0.78)
- **False Positives:** DNS wildcards inflate results; use resolver confirmation before web scanning
- **Takeover Risk:** Check CNAME records for orphaned/vulnerable subdomains
- **Handoff:** Live subdomains → `web-recon`, then `web-vuln`
- **Avoid:** Active DNS enumeration before confirming authorization; zone transfers on production domains
