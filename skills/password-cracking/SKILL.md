---
name: password-cracking
description: Hash identification, offline cracking, and credential attack workflows for HexStrike tools. Use when you have hashes, captured handshakes, or login services that need controlled password testing.
---

# Password Cracking

## When to use

Use this skill for:

- unknown hash identification
- offline cracking with `john` or `hashcat`
- credential brute-force against a specific service
- validating password reuse with approved scope

## Working Style

1. identify the format first with `hashid`
2. prefer offline cracking before network brute-force
3. keep brute-force targeted and rate-aware

Preferred entrypoint:

```python
hashid(hash="<paste_hash_here>")
```

See `REFERENCE.md` for common cracking and brute-force calls.

## Notes

- do not guess the hash type
- for web, SMB, or service-specific follow-up, combine this skill with the relevant recon skill
- if the user only needs enumeration of shares or auth methods, use `smb-enum` or `active-directory` first
