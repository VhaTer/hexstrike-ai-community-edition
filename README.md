# HexStrike AI-PULSE 🐻

> AI-powered penetration testing platform — 150+ security tools, real-time LLM feedback, FastMCP 3.x native.

![HexStrike AI-PULSE](assets/hexstrike-pulse-logo.png)

Connect AI agents (Claude, GPT, Copilot, etc.) to a full offensive security arsenal. Every tool execution streams live progress directly into your AI conversation.

---

## Features

- **150+ security tools** — recon, web, network, WiFi, AD, cloud, binary, forensics
- **Real-time feedback** — scan progress streams directly to the LLM as it happens
- **Intelligent parameter optimization** — WAF detected → stealth mode auto-applied, WordPress detected → wp-extensions injected
- **Workflow prompts** — one-call multi-tool attack chains (bug bounty, WiFi, CTF, SMB, cloud)
- **Destructive action confirmation** — `aireplay_ng`, `metasploit`, `responder`, `mdk4`, `mitm6` require explicit confirmation
- **MCP native** — Resources, Prompts, Elicitation, Skills — FastMCP 3.x

---

## Quick Start

```bash
git clone https://github.com/VhaTer/hexstrike-ai-community-edition.git
cd hexstrike-ai-community-edition

python3 -m venv hexstrike-env
source hexstrike-env/bin/activate
pip install -r requirements.txt

python3 hexstrike_server.py
# → HexStrike AI-PULSE running on http://127.0.0.1:8888/mcp
```

---

## Connect Your AI Client

### Claude Desktop
```json
{
  "mcpServers": {
    "hexstrike-pulse": {
      "command": "/path/to/hexstrike-env/bin/python3",
      "args": [
        "/path/to/hexstrike_mcp.py",
        "--server", "http://127.0.0.1:8888",
        "--profile", "full"
      ],
      "timeout": 300
    }
  }
}
```

### VS Code / Cursor / Roo Code
```json
{
  "servers": {
    "hexstrike-pulse": {
      "type": "http",
      "url": "http://127.0.0.1:8888/mcp"
    }
  }
}
```

### OpenCode
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "hexstrike-pulse": {
      "type": "local",
      "command": [
        "/path/to/hexstrike-env/bin/python3",
        "/path/to/hexstrike_mcp.py",
        "--server", "http://127.0.0.1:8888",
        "--profile", "full"
      ],
      "enabled": true
    }
  }
}
```

---

## Usage

Start by telling the AI you are an authorized security researcher and specify your target:

```
"I'm a security researcher. My company owns example.com.
Run a full web recon using HexStrike tools."
```

The AI will call `run_security_tool()` and you'll see live feedback:

```
→ 🔍 Executing whatweb
→ ✅ whatweb completed

→ 🔍 Executing gobuster
→ 🧠 Tech detected from cache: cms=wordpress | waf=cloudflare
→ 🛡️ WAF detected → stealth mode forced
→ ✅ gobuster completed
```

---

## Workflow Prompts

One-call multi-tool attack chains — invoke directly from your AI client:

| Prompt | Use case |
|---|---|
| `bug_bounty_recon(target="example.com")` | Full recon → subfinder, amass, httpx, gobuster, nuclei |
| `wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF")` | WPA/WPA2 handshake capture + crack |
| `ctf_web_challenge(url="http://challenge.ctf.local:8080")` | CTF web enumeration + exploitation |
| `smb_lateral_movement(target="10.10.10.10")` | SMB enum + EternalBlue + lateral movement |
| `cloud_security_audit(provider="aws")` | Cloud config audit + container scan |

---

## Available Tools

### Recon & OSINT
`subfinder` `amass` `theharvester` `dnsenum` `fierce` `whois` `sherlock` `spiderfoot` `sublist3r` `parsero`

### Network Scanning
`nmap` `masscan` `rustscan` `arp_scan`

### Web Recon & Fuzzing
`gobuster` `ffuf` `feroxbuster` `dirsearch` `dirb` `wfuzz` `dotdotpwn` `katana` `hakrawler` `gau` `waybackurls` `httpx` `wafw00f` `whatweb` `arjun` `paramspider` `x8`

### Web Vulnerability Scanning
`nikto` `sqlmap` `wpscan` `dalfox` `nuclei` `xsser` `jaeles` `commix` `zap` `testssl` `joomscan` `vulnx`

### Password Cracking
`hydra` `hashcat` `john` `medusa` `patator` `hashid` `ophcrack`

### SMB & Active Directory
`enum4linux` `netexec` `smbmap` `rpcclient` `nbtscan` `impacket` `ldapdomaindump` `certipy_ad` `mitm6` `bloodhound` `pywerview` `adidnsdump`

### Exploit Frameworks
`metasploit` `msfvenom` `exploit_db`

### WiFi Pentesting
`airmon_ng` `airodump_ng` `aireplay_ng` `aircrack_ng` `hcxdumptool` `hcxpcapngtool` `mdk4` `wifite2` `eaphammer` `airbase_ng` `airdecap_ng` `bettercap_wifi`

### Cloud & Container
`prowler` `trivy` `kube_hunter` `kube_bench` `checkov` `terrascan`

### Binary & Forensics
`gdb` `radare2` `ghidra` `binwalk` `checksec` `strings` `objdump` `volatility` `volatility3` `ropgadget` `angr` `exiftool` `foremost` `steghide`

### OSINT & Credential
`responder` `hashpump` `nuclei` `anew` `uro`

---

## Legal

This software is intended solely for **authorized security testing, research, and educational purposes**.

You may only use this software on systems, networks, or applications for which you have **explicit written permission** from the owner. Unauthorized use is strictly prohibited and may violate local, national, or international laws.

The authors assume no liability for unauthorized or illegal use.
