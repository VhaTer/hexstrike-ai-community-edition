---
name: hexstrike-workflows
description: HexStrike AI native workflow prompts — invoke full multi-tool attack chains with a single prompt call
---

# hexstrike-workflows

HexStrike AI native workflow prompts. Use these when a user asks to run a complete engagement workflow, not just a single tool. Each prompt chains multiple `run_security_tool()` calls in the correct order.

## Available workflows

| Prompt | Use case | Key tools |
|---|---|---|
| `bug_bounty_recon(target)` | Full bug bounty recon on a domain | subfinder, amass, httpx, nmap, gobuster, nuclei |
| `wifi_attack_chain(interface, bssid)` | WPA/WPA2 handshake capture + crack | airmon_ng, airodump_ng, aireplay_ng, aircrack_ng |
| `ctf_web_challenge(url)` | CTF web challenge enumeration + exploitation | gobuster, nuclei, sqlmap, dalfox, dotdotpwn |
| `smb_lateral_movement(target)` | SMB enumeration + lateral movement | enum4linux, smbmap, netexec, metasploit |
| `cloud_security_audit(provider)` | Cloud config audit + container scan | prowler, trivy, kube_hunter |

## Syntax
```
bug_bounty_recon(target="example.com")
wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF", channel="6")
ctf_web_challenge(url="http://challenge.ctf.local:8080")
smb_lateral_movement(target="10.10.10.10")
cloud_security_audit(provider="aws", profile="default")
```

## Tool syntax inside prompts

All prompts use the Phase 3 direct execution interface:
```
run_security_tool(tool_name="<tool>", parameters='{"key": "value"}')
```

## Destructive action warnings

Prompts containing `aireplay_ng` deauth or `metasploit` exploit include explicit `⚠️ User confirmation required` messages — these trigger `ctx.elicit()` in Claude Desktop.

## Skills used by each workflow

| Prompt | Skills referenced |
|---|---|
| `bug_bounty_recon` | subdomain-enum, nmap-recon, web-recon, web-vuln |
| `wifi_attack_chain` | wifi_pentest |
| `ctf_web_challenge` | web-recon, web-vuln |
| `smb_lateral_movement` | smb-enum, exploitation |
| `cloud_security_audit` | cloud-audit |

## HexStrike Tool Reference

| Tool | DIRECT_TOOLS key | Workflow |
|---|---|---|
| `subfinder` | `subfinder` | bug_bounty_recon |
| `amass` | `amass` | bug_bounty_recon |
| `httpx` | `httpx` | bug_bounty_recon, ctf_web_challenge |
| `rustscan` | `rustscan` | bug_bounty_recon |
| `nmap` | `nmap` | bug_bounty_recon, smb_lateral_movement |
| `wafw00f` | `wafw00f` | bug_bounty_recon, ctf_web_challenge |
| `katana` | `katana` | bug_bounty_recon, ctf_web_challenge |
| `gobuster` | `gobuster` | bug_bounty_recon, ctf_web_challenge |
| `nuclei` | `nuclei` | bug_bounty_recon, ctf_web_challenge |
| `nikto` | `nikto` | bug_bounty_recon, ctf_web_challenge |
| `airmon_ng` | `airmon_ng` | wifi_attack_chain |
| `airodump_ng` | `airodump_ng` | wifi_attack_chain |
| `aireplay_ng` | `aireplay_ng` | wifi_attack_chain |
| `aircrack_ng` | `aircrack_ng` | wifi_attack_chain |
| `ffuf` | `ffuf` | ctf_web_challenge |
| `sqlmap` | `sqlmap` | ctf_web_challenge |
| `dalfox` | `dalfox` | ctf_web_challenge |
| `dotdotpwn` | `dotdotpwn` | ctf_web_challenge |
| `nbtscan` | `nbtscan` | smb_lateral_movement |
| `enum4linux` | `enum4linux` | smb_lateral_movement |
| `smbmap` | `smbmap` | smb_lateral_movement |
| `netexec` | `netexec` | smb_lateral_movement |
| `metasploit` | `metasploit` | smb_lateral_movement |
| `prowler` | `prowler` | cloud_security_audit |
| `trivy` | `trivy` | cloud_security_audit |
| `kube_hunter` | `kube_hunter` | cloud_security_audit |
