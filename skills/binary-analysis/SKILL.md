---
name: binary-analysis
description: Binary triage, reverse engineering, and exploit gadget discovery via static analysis and debuggers. Use for local binary assessment and ROP/payload development.
---

# Binary Analysis

## When to use

Use this skill when the target is a **local binary, firmware, or exploit artifact** rather than live service:

- Security mitigation assessment (NX, ASLR, PIE, Canary)
- String and symbol extraction (hardcoded creds, debug info, library paths)
- Static disassembly and control flow analysis
- ROP gadget hunting and payload development
- Firmware extraction and carving

## Working Style

**Staged analysis prevents wasted reversing effort:**

1. **Triage** — `checksec` (mitigations), `file`, `strings` (obvious artifacts)
2. **Quick Win** — Extract hardcoded strings, debug symbols, embedded credentials
3. **Static Analysis** — `radare2` or `ghidra` for control flow; focus on dangerous functions (system(), memcpy)
4. **Gadget Hunting** — `ropper` or `ropgadget` if binary lacks modern mitigations; assemble ROP chains
5. **Debug Verification** — `gdb` to confirm payload behavior (only in lab environment)

**Entry point:**

```python
checksec(file="/path/to/binary")
```

## Notes

- **Effectiveness:** checksec (1.0), strings (0.95), ropper (0.90), ghidra (0.92)
- **Mitigations:** ASLR reduces gadget viability; bypass via info leak or heap spray
- **Time Investment:** Reversing is time-intensive; prioritize quick wins (strings, symbols) first
- **Handoff:** Confirmed gadgets and payload → `exploitation` for delivery/shell execution
- **Avoid:** Deep reversing before confirming mitigations; CTF-style techniques on production binaries without incident response plan
