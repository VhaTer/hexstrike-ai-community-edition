---
name: smb-enum
description: SMB service enumeration, share discovery, and RPC interrogation on Windows networks. Use to map Windows hosts before credential testing or lateral movement.
---

# SMB Enum

## When to use

Use this skill in **Phase 2: Enumeration** on identified Windows hosts (ports 139, 445):

- SMB service enumeration (version, dialect, signing requirement)
- Share enumeration and null-session access validation
- User/group/policy enumeration via RPC/LDAP
- Domain membership and local admin detection
- Relay and coercion attack surface assessment

## Working Style

**Staged enumeration minimizes detection and lockout risk:**

1. **Exposure Check** — `smbmap` (null/guest access), `enum4linux -a` (full recon) [critical, ~2-5 min]
2. **Share Access** — Test null/guest access; enumerate readable shares (information disclosure) [high, ~2-5 min]
3. **User Enum** — RPC users/groups; LDAP if domain-joined → handoff to `active-directory` [high, ~30-60s]
4. **Cred Testing** — Only after user discovery; pair with `password-cracking` if cred list exists [high, ~2-10 min]
5. **Exploit Prep** — Identify services vulnerable to relay/coercion before triggering [critical, ~30-60s]

**Decision fields (aligned with smb_lateral_movement prompt):**
- **result_context:** After `nmap --script smb-vuln-ms17-010`, check "VULNERABLE" vs "NOT VULNERABLE" — determines Path A (EternalBlue exploitation) vs Path B (credential brute-force).
- **result_context:** After `nmap smb-security-mode`, check SMB signing status — disabled means relay attacks are viable.
- **result_context:** After `smbmap`, check for writable shares — potential SCF/shortcut hijack vector.
- **Fallback:** If EternalBlue patched, pivot to hydra brute-force with rockyou.txt + common usernames.

**Entry point:**

```python
enum4linux(target="10.10.10.10", actions="-a")
```

## Notes

- **Effectiveness:** enum4linux (0.88), smbmap (0.90), netexec (0.92)
- **Detection:** SMB scanning is logged by Windows Defender; assume blue team awareness
- **Domain Detection:** If domain-joined, transition to `active-directory` for LDAP/Kerberos attacks
- **Relay Risk:** Check SMB signing requirement; unsigned = relay vulnerability (requires `responder`)
- **Avoid:** Aggressive share enumeration without scope; credential stuffing without rate limiting
