---
name: binary-analysis
description: Binary triage, reversing, and exploit-development workflows for HexStrike tools. Use when you need mitigation checks, string extraction, firmware carving, static analysis, gadget hunting, or debugger-assisted triage.
---

# Binary Analysis

## When to use

Use this skill when the target is a local binary, firmware image, or exploit-development artifact rather than a live network service.

## Working Style

1. start with triage tools such as `checksec`, `strings`, or `binwalk`
2. move to `radare2`, `gdb`, or gadget search only when the binary warrants it
3. keep the outcome clear: triage, reverse engineering, or exploit construction

Preferred entrypoint:

```python
checksec(file="/path/to/binary")
```

See `REFERENCE.md` for tool-specific calls.

## Notes

- use this skill for files and reversing work, not for live exploitation of services
- if the task becomes payload generation or shell delivery, switch to `exploitation`
