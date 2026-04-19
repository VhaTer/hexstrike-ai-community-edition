---
name: osint-recon
description: Open-source intelligence gathering on people, organizations, and domains via public sources and historical records.
---

# OSINT Recon

## When to use

Use this skill for **pre-engagement recon** using only public sources:

- Username harvesting (social media, GitHub, forums)
- Organization footprint (domains, email patterns, employees)
- Domain/IP WHOIS, historic DNS, and SSL certificate history
- Exposed data (breaches, pastes, unindexed caches)
- Email address validation and MX record enumeration

## Working Style

**Passive-only prevents triggering monitoring:**

1. **Social Media** — `sherlock` or `bbot` (username search across platforms)
2. **Email Harvesting** — `theharvester` (Google/Shodan/Censys dorks)
3. **Domain History** — WHOIS, `wayback_urls` (archived pages), SSL cert history
4. **Breach Search** — https://breachdirectory.org, Have I Been Pwned API (no direct tooling)
5. **Validation** — `validate_email` to confirm working addresses without probing
6. **Handoff** — Confirmed domain/org → `subdomain-enum`, then `web-recon`

**Entry point:**

```python
sherlock(username="targetuser", verbose=False)
```

## Notes

- **Effectiveness:** sherlock (0.88), theharvester (0.85), bbot (0.92)
- **Detection:** Passive only; no active probing in OSINT phase (avoid triggering WAF, IDS)
- **Data Sources:** Results vary by API availability (Shodan key, Censys key, etc.)
- **No Email Verification:** Do NOT send emails or probe for deliverability (triggers spam filters, IDS)
- **Handoff:** Confirmed org/domain → `subdomain-enum` for asset discovery
- **Avoid:** Active probing (port scan, HTTP requests) during OSINT phase; credential guessing
