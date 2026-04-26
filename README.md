# HexStrike AI-PULSE

> AI-powered penetration testing platform — 150+ security tools, real-time LLM feedback, MCP native.

![HexStrike AI-PULSE](assets/hexstrike-pulse-logo.png)

Connect any AI agent (Claude, GPT, Copilot, and more) to a full offensive security arsenal. Every tool execution streams live progress directly into your AI conversation — the AI sees what's happening, suggests next steps, and asks for confirmation before running destructive actions.

---

## Features

- **150+ security tools** — recon, web, network, WiFi, Active Directory, cloud, binary analysis, forensics
- **Real-time streaming** — scan progress, results, and intelligent suggestions flow directly to the LLM as they happen
- **Intelligent attack planning** — `plan_attack()` analyzes a target and generates an ordered attack chain with tool selection and success probabilities
- **Smart parameter tuning** — WAF detected → stealth mode applied automatically, WordPress detected → relevant extensions injected
- **Workflow prompts** — one-call multi-tool attack sequences for bug bounty, WiFi, CTF, SMB lateral movement, and cloud audits
- **Safety gates** — `aireplay-ng`, `metasploit`, `responder`, `mdk4`, `mitm6` require explicit confirmation before execution
- **Skill guidance** — before each tool runs, the AI receives operational best-practice guidance for that tool
- **MCP native** — built on FastMCP 3.x with Resources, Prompts, Elicitation, and Context streaming

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

<details>
<summary><b>Claude Desktop / Claude.ai</b></summary>

```json
{
  "servers": {
    "hexstrike-ai": {
      "url": "http://127.0.0.1:8888/mcp",
      "type": "http"
    }
  }
}
```

</details>

<details>
<summary><b>VS Code / Cursor / Roo Code</b></summary>

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

</details>

<details>
<summary><b>OpenCode</b></summary>

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "hexstrike-pulse": {
      "type": "http",
      "url": "http://127.0.0.1:8888/mcp",
      "enabled": true
    }
  }
}
```

</details>

---

## Usage

Tell the AI you are an authorized security researcher and specify your target:

```
"I'm a security researcher. My company owns example.com.
Run a full web recon using HexStrike tools."
```

The AI streams live feedback as tools execute:

```
→ 🔍 Executing whatweb
→ 📚 [web-recon] Web Technology Identification — use before any targeted attack
→ ✅ whatweb completed

→ 🔍 Executing gobuster
→ 🧠 Tech detected: cms=wordpress | waf=cloudflare
→ 🛡️ WAF detected → stealth mode forced
→ ✅ gobuster completed
```

### Attack Planning

Ask the AI to plan before executing:

```
"Plan an attack against 10.10.10.10 for a CTF engagement."
```

```
→ 🧠 Analyzing target: 10.10.10.10 (objective=ctf)
→ 🎯 Target type: linux_server | Risk: high | Confidence: 87%
→ ✅ Attack chain ready: 8 steps | Est. time: 420s | P(success): 73%
```

---

## Workflow Prompts

One-call multi-tool attack sequences — invoke directly from your AI client:

| Prompt | Use case |
|---|---|
| `bug_bounty_recon(target="example.com")` | Full recon → subfinder, amass, httpx, gobuster, nuclei |
| `wifi_attack_chain(interface="wlan0", bssid="AA:BB:CC:DD:EE:FF")` | WPA/WPA2 handshake capture and crack |
| `ctf_web_challenge(url="http://challenge.ctf.local:8080")` | CTF web enumeration and exploitation |
| `smb_lateral_movement(target="10.10.10.10")` | SMB enumeration and lateral movement |
| `cloud_security_audit(provider="aws")` | Cloud configuration audit and container scan |

---

## Available Tools

<details>
<summary><b>🔍 Network Reconnaissance & Scanning</b></summary>

- **Nmap** — Port scanning with service detection and NSE scripts
- **Rustscan** — Ultra-fast port scanner
- **Masscan** — High-speed Internet-scale port scanning
- **AutoRecon** — Automated multi-tool reconnaissance
- **Amass** — Subdomain enumeration and OSINT
- **Subfinder** — Fast passive subdomain discovery
- **Fierce** — DNS reconnaissance and zone transfer testing
- **DNSEnum** — DNS information gathering
- **TheHarvester** — Email and subdomain harvesting
- **ARP-Scan** — Network discovery via ARP
- **NBTScan** — NetBIOS name scanning
- **RPCClient** — RPC enumeration
- **Whois** — Domain and IP registration lookup
- **Enum4linux / Enum4linux-ng** — SMB enumeration
- **SMBMap** — SMB share enumeration and exploitation
- **Responder** — LLMNR/NBT-NS/MDNS poisoner
- **NetExec** — Network service exploitation framework

</details>

<details>
<summary><b>📡 WiFi Penetration Testing</b></summary>

- **Aircrack-ng suite** — Monitor mode, packet capture, deauth, WPA cracking
- **hcxdumptool / hcxpcapngtool** — Clientless PMKID capture
- **EAPHammer** — WPA-Enterprise Evil Twin attacks
- **Wifite2** — Automated WiFi auditing
- **Bettercap** — WiFi recon and Evil Twin
- **mdk4** — 802.11 protocol stress testing

</details>

<details>
<summary><b>🌐 Web Application Security</b></summary>

- **Gobuster / Dirsearch / Feroxbuster / FFuf** — Directory and parameter fuzzing
- **HTTPx / Katana / Hakrawler** — HTTP probing, crawling, endpoint discovery
- **Nuclei** — Vulnerability scanner with 4000+ templates
- **Nikto** — Web server vulnerability scanner
- **SQLMap** — SQL injection testing
- **WPScan** — WordPress security scanner
- **Dalfox** — XSS vulnerability scanning
- **Wafw00f** — WAF fingerprinting
- **TestSSL / SSLScan / SSLyze** — SSL/TLS assessment
- **Whatweb** — Web technology identification
- **JWT-Tool** — JSON Web Token testing
- **Commix** — Command injection exploitation
- **ZAP / Burp Suite** — Proxy-based web testing

</details>

<details>
<summary><b>🔐 Authentication & Password Security</b></summary>

- **Hydra / Medusa / Patator** — Network login brute-forcing
- **Hashcat** — GPU-accelerated password recovery
- **John the Ripper** — Password hash cracking
- **Evil-WinRM** — Windows Remote Management shell
- **HashID** — Hash algorithm identification
- **NetExec** — Post-exploitation and lateral movement

</details>

<details>
<summary><b>🏢 Active Directory</b></summary>

- **BloodHound / SharpHound** — AD attack path mapping
- **Impacket suite** — SMB, Kerberos, DCOM attacks
- **Kerbrute** — Kerberos user enumeration and brute-forcing
- **LDAPDomainDump** — Active Directory LDAP enumeration
- **Responder** — Credential harvesting via LLMNR/NBT-NS
- **mitm6** — IPv6 DNS takeover
- **CrackMapExec / NetExec** — Swiss army knife for AD pentesting

</details>

<details>
<summary><b>🔬 Binary Analysis & Reverse Engineering</b></summary>

- **GDB + PEDA/GEF** — Debugger with exploit development extensions
- **Radare2 / Ghidra** — Reverse engineering frameworks
- **Binwalk** — Firmware analysis and extraction
- **ROPgadget / Ropper** — ROP chain building
- **Pwntools** — CTF exploit development framework
- **Checksec** — Binary security property checker
- **Volatility / Volatility3** — Memory forensics

</details>

<details>
<summary><b>☁️ Cloud & Container Security</b></summary>

- **Prowler** — AWS/Azure/GCP security assessment
- **Scout Suite** — Multi-cloud security auditing
- **Pacu** — AWS exploitation framework
- **Trivy** — Container and IaC vulnerability scanner
- **Kube-Hunter / Kube-Bench** — Kubernetes security testing
- **Checkov / Terrascan** — Infrastructure as code scanning

</details>

<details>
<summary><b>🕵️ OSINT & Bug Bounty</b></summary>

- **Sherlock / Social-Analyzer** — Username and social media investigation
- **SpiderFoot / Recon-ng / Maltego** — OSINT automation and link analysis
- **Shodan / Censys** — Internet-connected asset discovery
- **TruffleHog** — Git secret scanning
- **Aquatone** — Visual website inspection across hosts
- **Subjack** — Subdomain takeover checker

</details>

<details>
<summary><b>🏆 CTF & Forensics</b></summary>

- **Steghide / Stegsolve / Zsteg** — Steganography detection and extraction
- **Foremost / Scalpel / PhotoRec** — File carving and recovery
- **ExifTool** — Metadata analysis
- **Autopsy / Sleuth Kit** — Digital forensics platform
- **CyberChef** — Encoding, encryption, and data analysis

</details>

---

## Legal

This software is intended solely for **authorized security testing, research, and educational purposes**.

You may only use this software on systems, networks, or applications for which you have **explicit written permission** from the owner. Unauthorized use is strictly prohibited and may violate local, national, or international laws.

The authors assume no liability for unauthorized or illegal use.
