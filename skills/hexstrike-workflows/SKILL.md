---
name: hexstrike-workflows
description: Native HexStrike workflow prompts and compound playbooks. Use when the user wants a full engagement sequence rather than a single tool execution.
---

# HexStrike Workflows

## When to use

Use this skill when the request is sequence-shaped:

- "run a full bug bounty recon flow"
- "do the Wi-Fi capture chain"
- "audit the cloud estate end to end"
- "walk the SMB path from enum to exploit readiness"

## Working Style

1. prefer the registered MCP prompts for opinionated multi-step flows
2. if a prompt is not suitable, chain `run_security_tool(...)` deliberately and explain each phase
3. pull detailed per-tool guidance from the mapped skills with `get_tool_skill(tool_name=...)`

## Available Prompt Families

- `bug_bounty_recon`
- `wifi_attack_chain`
- `ctf_web_challenge`
- `smb_lateral_movement`
- `cloud_security_audit`

See `REFERENCE.md` for prompt names, linked skills, and tool entrypoints.
