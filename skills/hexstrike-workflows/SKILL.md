---
name: hexstrike-workflows
description: Curated MCP prompts for multi-step attack chains and engagement playbooks. Use for guided workflows spanning multiple skills. Each prompt step now carries expected_time, priority, fallback, and result_context for agent-guided execution.
---

# HexStrike Workflows

## When to use

Use this skill when the request spans multiple phases and tools:

- "Full bug bounty recon workflow"
- "WiFi penetration testing chain"
- "Cloud infrastructure security audit"
- "SMB lateral movement path"
- "CTF web challenge walk-through"
- "Universal CTF challenge (any category)"

## Working Style

**Each registered prompt step includes four decision-making fields:**

| Field | Purpose |
|---|---|
| `[priority: critical/high/medium/low]` | Execution order — process critical first, skip low if time-constrained |
| `[expected: 10s-60min]` | Rough time estimate per step (guides agent planning and timeout setting) |
| `→ Fallback` | Alternative tool or approach if the primary tool fails |
| `→ result_context` | Expected output signal to check before proceeding to next step |

**Execution flow:**

1. **Invoke Prompt** — Call registered MCP prompt for opinionated multi-tool sequence
2. **Check priority** — Process critical/high steps first; skip low-priority if target hardened
3. **Check result_context** — After each tool: did it find what the step expected? If not, follow the fallback
4. **Tool Execution** — HexStrike chains typed MCP tools with automatic tech detection and parameter optimization
5. **Per-Tool Detail** — Use `get_tool_skill(tool_name=...)` for deep guidance on specific steps
6. **Custom Chains** — For non-standard flows, manually sequence skills (e.g., custom pivot path)

## Registered Prompts

- `bug_bounty_recon` — Passive OSINT → subdomain enum → web recon → vuln scan (no exploitation). ~15 min estimated, 10 tools, critical-first priority ordering.
- `wifi_attack_chain` — AP discovery → handshake capture → cracking → network pivot. ~5 min, 4 tools, deauth requires confirmation.
- `ctf_web_challenge` — Web recon → targeted vuln scan → exploit lookup → payload crafting. Powered by CTFWorkflowManager V6.
- `ctf_challenge` — Universal CTF across all 7 categories (web/crypto/pwn/forensics/rev/misc/osint). Powered by CTFWorkflowManager V6.
- `smb_lateral_movement` — SMB enum → password cracking → AD enumeration → coercion/delegation abuse. ~15 min, 6 tools, EternalBlue decision point.
- `cloud_security_audit` — Cloud account review → container scanning → Kubernetes RBAC assessment. ~60 min estimated, 3 tools, prowler credential fallback.

**Entry point:**

```python
# Prompt invocation (handled via MCP)
# E.g., "Use bug_bounty_recon prompt for example.com"
```

## Notes

- **Effectiveness:** Prompts integrate multiple tools with effectiveness 0.85–0.93
- **Tech Detection:** Automatic; flows adapt to WordPress, ASP.NET, cloud providers, etc.
- **Multi-Phase:** Each prompt maps to standard penetration testing phases (recon → enum → assess → exploit)
- **Decision Points:** Some prompts (e.g., smb_lateral_movement STEP 3) branch on tool output ("VULNERABLE" vs "NOT VULNERABLE"). Check result_context to choose path.
- **Customization:** For non-standard targets, manually sequence skills instead of prompts
- **Safety:** Destructive tools (aireplay_ng, responder, metasploit) require confirmation even in prompts
- **Avoid:** Mixing unrelated skills in custom chains; always maintain phase ordering to minimize noise
