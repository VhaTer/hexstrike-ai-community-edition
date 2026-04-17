---
name: smb-enum
description: SMB and Windows share enumeration workflows for HexStrike tools. Use when you need host information, shares, auth testing, RPC insight, or lateral movement preparation on Windows networks.
---

# SMB Enum

## When to use

Use this skill for Windows-network discovery and low-impact SMB follow-up before exploitation.

## Working Style

1. identify basic SMB exposure and host info
2. enumerate shares, users, sessions, and policy details
3. test credentials only after you have a clear target account strategy

Preferred entrypoint:

```python
enum4linux(target="10.10.10.10")
```

See `REFERENCE.md` for the common calls.

## Notes

- if the environment is domain-joined and the user needs AD graphing or certificate abuse follow-up, switch to `active-directory`
- if the next step is confirmed exploitation, switch to `exploitation`
