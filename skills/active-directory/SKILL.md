---
name: active-directory
description: Active Directory reconnaissance and abuse-path preparation workflows for HexStrike AD tools. Use when the target is a Windows domain and you need LDAP, DNS, certificate, graph, or coercion-related visibility.
---

# Active Directory

## When to use

Use this skill when the environment is domain-based and simple SMB enumeration is no longer enough.

## Working Style

1. collect domain structure and identities first
2. map certificate, DNS, and graph exposures next
3. keep poisoning or relay-adjacent steps explicit and scoped

Preferred entrypoint:

```python
ldapdomaindump(hostname="dc01.example.local", username="user@example.local", password="Passw0rd!")
```

See `REFERENCE.md` for common AD tool calls.
