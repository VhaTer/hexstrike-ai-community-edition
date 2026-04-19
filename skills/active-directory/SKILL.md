---
name: active-directory
description: Active Directory reconnaissance, privilege escalation path analysis, and Kerberos attack surface mapping. Use for domain exploitation and lateral movement.
---

# Active Directory

## When to use

Use this skill in **Phase 2-5: Enumeration through Post-Exploitation** on domain-joined networks:

- Domain structure, trust, and user/group enumeration
- Certificate authority enumeration and ESC exploitation
- Kerberos delegation abuse (constrained, unconstrained, RBCD)
- Relay and coercion attack surface (NTLM relay, PrinterBug, PetitPotam)
- Lateral movement path identification (ACL chains, group memberships)

## Working Style

**Staged targeting reduces noise and EDR/detection:**

1. **LDAP Enum** — `ldapdomaindump` or `netexec ldap` (unauthenticated or with null creds)
2. **Domain Trust** — Identify domain trusts, forest relationships, external trusts
3. **Certificate Enum** — `certipy find` for CA exposure and ESC paths
4. **Delegation Abuse** — Check `msDS-AllowedToDelegateTo` for constrained delegation, `msFLAG_ACCOUNTS_TRUSTED_FOR_DELEGATION` for unconstrained
5. **Coercion Surface** — Enumerate relayable services (Exchange, cert auth, Web enrollment)

**Entry point:**

```python
ldapdomaindump(hostname="dc01.example.local", username="user@example.local", password="Passw0rd!", start_tls=True)
```

## Notes

- **Prerequisites:** Always use SMB enum first; AD requires domain context
- **Effectiveness:** ldapdomaindump (0.95), certipy (0.93), netexec (0.92)
- **Detection:** LDAP queries are logged; assume blue team visibility
- **Relay Risk:** Unsigned SMB + unconstrained delegation = high-value target
- **Handoff:** Confirmed privilege escalation path → `exploitation` or `password-cracking` (for unconstrained/RBCD)
- **Avoid:** Unauthenticated LDAP queries on hardened domains; triggering coercion without mitigation plan
