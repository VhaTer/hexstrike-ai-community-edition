---
name: subdomain-enum
description: Domain and subdomain discovery workflows for HexStrike recon tools. Use when the target is an internet-facing domain and you need asset expansion before web or network follow-up.
---

# Subdomain Enum

## When to use

Use this skill for passive-first asset discovery on a domain:

- subdomain collection
- DNS enumeration
- certificate and public source harvesting
- expanding a bug bounty or external attack surface

## Working Style

Sequence the work:

1. start with `subfinder` and `amass`
2. add `theharvester`, `dnsenum`, or `fierce` when you need broader DNS context
3. probe live hosts afterward with `httpx` or feed results into `web-recon`

Prefer:

```python
subfinder(domain="example.com")
```

See `REFERENCE.md` for the main tool calls.

## Notes

- keep passive enumeration first when stealth matters
- normalise and deduplicate before downstream web scanning
- if the request is really OSINT on people, usernames, or org metadata, use `osint-recon` instead
