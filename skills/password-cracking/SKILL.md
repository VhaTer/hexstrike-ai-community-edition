---
name: password-cracking
description: Hash identification, offline cracking, and credential testing against services. Use to recover passwords from captured hashes or validate weak credentials.
---

# Password Cracking

## When to use

Use this skill when you have:

- Captured hashes (NTLM, SHA, bcrypt, WordPress, etc.)
- WiFi handshakes (WPA2/WPA3 PMKID/EAPOL)
- Weak credentials to test against services (SMB, SSH, HTTP)
- Password spraying opportunity (low lockout threshold)

## Working Style

**Always offline before online; avoid account lockout:**

1. **Identify** — `hashid` on unknown format; confirm via hash characteristics (length, prefix)
2. **Crack Offline** — `hashcat` (GPU fast) or `john` (CPU/rule-based); prefer wordlist + rules over bruteforce
3. **Online Testing** — `hydra` or `medusa` with rate limiting; confirm target lockout policy first
4. **Validation** — Confirm creds on actual service before moving to exploitation

**Entry point:**

```python
hashid(hash="5d41402abc4b2a76b9719d911017c592")
```

## Notes

- **Prerequisites:** Never guess hash type; `hashid` eliminates misidentification
- **Effectiveness:** hashcat (0.95 GPU), john (0.85), hydra (0.80 with rate limiting)
- **Lockout Risk:** Check account lockout policy; use `--spraying` mode (low, distributed attempts)
- **Handoff:** Confirmed creds → `smb-enum`, `active-directory`, or `exploitation`
- **Avoid:** Online brute-force without prior service enumeration; dictionary attacks on protected services without frequency analysis
