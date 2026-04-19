---
name: hexstrike-workflows
description: Curated MCP prompts for multi-step attack chains and engagement playbooks. Use for guided workflows spanning multiple skills.
---

# HexStrike Workflows

## When to use

Use this skill when the request spans multiple phases and tools:

- "Full bug bounty recon workflow"
- "WiFi penetration testing chain"
- "Cloud infrastructure security audit"
- "SMB lateral movement path"
- "CTF web challenge walk-through"

## Working Style

**Registered MCP prompts automate sequencing and tool selection:**

1. **Invoke Prompt** — Call registered MCP prompt for opinionated multi-tool sequence
2. **Tool Execution** — HexStrike chains `run_security_tool()` calls with automatic tech detection and parameter optimization
3. **Per-Tool Detail** — Use `get_tool_skill(tool_name=...)` for deep guidance on specific steps
4. **Custom Chains** — For non-standard flows, manually sequence skills (e.g., custom pivot path)

## Registered Prompts

- `bug_bounty_recon` — Passive OSINT → subdomain enum → web recon → vuln scan (no exploitation)
- `wifi_attack_chain` — AP discovery → handshake capture → cracking → network pivot
- `ctf_web_challenge` — Web recon → targeted vuln scan → exploit lookup → payload crafting
- `smb_lateral_movement` — SMB enum → password cracking → AD enumeration → coercion/delegation abuse
- `cloud_security_audit` — Cloud account review → container scanning → Kubernetes RBAC assessment

**Entry point:**

```python
# Prompt invocation (handled via MCP)
# E.g., "Use bug_bounty_recon prompt for example.com"
```

## Notes

- **Effectiveness:** Prompts integrate multiple tools with effectiveness 0.85–0.93
- **Tech Detection:** Automatic; flows adapt to WordPress, ASP.NET, cloud providers, etc.
- **Multi-Phase:** Each prompt maps to standard penetration testing phases (recon → enum → assess → exploit)
- **Customization:** For non-standard targets, manually sequence skills instead of prompts
- **Safety:** Destructive tools (aireplay_ng, responder, metasploit) require confirmation even in prompts
- **Avoid:** Mixing unrelated skills in custom chains; always maintain phase ordering to minimize noise
