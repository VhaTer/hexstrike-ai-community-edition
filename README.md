# HexStrike-AI PULSE

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
# → HexStrike-AI PULSE running on http://127.0.0.1:8888/mcp
```

---

## Connect Your AI Client

<details>
<summary><b>Claude Desktop</b></summary>

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

</details>

<details> 
<summary><b> VS Code / Cursor / Roo Code</b></summary>

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

</details>
---

## Usage

Start by telling the AI you are an authorized security researcher and specify your target:

```md
"I'm a security researcher. My company owns example.com.
Run a full web recon using HexStrike tools."
```

The AI will call `run_security_tool()` and you'll see live feedback:

```md
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

</details>

<details>
<summary><b>🔍 Network Reconnaissance & Scanning</b></summary>

- **Nmap** - Advanced port scanning with custom NSE scripts and service detection
- **Rustscan** - Ultra-fast port scanner with intelligent rate limiting
- **Masscan** - High-speed Internet-scale port scanning with banner grabbing
- **AutoRecon** - Comprehensive automated reconnaissance with 35+ parameters
- **Amass** - Advanced subdomain enumeration and OSINT gathering
- **Subfinder** - Fast passive subdomain discovery with multiple sources
- **Fierce** - DNS reconnaissance and zone transfer testing
- **DNSEnum** - DNS information gathering and subdomain brute forcing
- **TheHarvester** - Email and subdomain harvesting from multiple sources
- **ARP-Scan** - Network discovery using ARP requests
- **NBTScan** - NetBIOS name scanning and enumeration
- **RPCClient** - RPC enumeration and null session testing
- **Whois** - Domain and IP registration lookup for ownership and OSINT
- **Enum4linux** - SMB enumeration with user, group, and share discovery
- **Enum4linux-ng** - Advanced SMB enumeration with enhanced logging
- **SMBMap** - SMB share enumeration and exploitation
- **Responder** - LLMNR, NBT-NS and MDNS poisoner for credential harvesting
- **NetExec** - Network service exploitation framework (formerly CrackMapExec)

</details>

<details>
<summary><b>📡 WiFi Penetration Testing</b></summary>

- Aircrack-ng Suite:
- Aircrack-ng - WPA/WPA2 PSK cracking from captured handshakes using dictionary attacks
- Airmon-ng - Enable/disable monitor mode and kill interfering processes
- Airodump-ng - Passive 802.11 packet capture for AP discovery and WPA handshake collection
- Aireplay-ng - Packet injection for deauthentication, fake authentication, and ARP replay attacks
- Airbase-ng - Rogue/soft access point creation for Evil Twin and client capture attacks
- Airdecap-ng - Decrypt WEP/WPA/WPA2 encrypted pcap capture files

*Modern WiFi Tools:*

- hcxdumptool - Clientless PMKID capture and WPA/WPA2 handshake collection (v7.0.0+)
- hcxpcapngtool - Convert hcxdumptool pcapng output to hashcat -m 22000 format
- EAPHammer - WPA-Enterprise Evil Twin for harvesting 802.1X EAP credentials
- Wifite2 - Automated WiFi auditing with PMKID, handshake, and WPS attack support
- Bettercap - WiFi recon, deauthentication, and Evil Twin via Bettercap wifi module
- mdk4 - 802.11 protocol stress testing and WIDS/WIPS evasion validation

</details>

<details>
<summary><b>🌐 Web Application Security Testing</b></summary>

- **Gobuster** - Directory, file, and DNS enumeration with intelligent wordlists
- **Dirsearch** - Advanced directory and file discovery with enhanced logging
- **Feroxbuster** - Recursive content discovery with intelligent filtering
- **FFuf** - Fast web fuzzer with advanced filtering and parameter discovery
- **Dirb** - Comprehensive web content scanner with recursive scanning
- **HTTPx** - Fast HTTP probing and technology detection
- **Katana** - Next-generation crawling and spidering with JavaScript support
- **Hakrawler** - Fast web endpoint discovery and crawling
- **Gau** - Get All URLs from multiple sources (Wayback, Common Crawl, etc.)
- **Waybackurls** - Historical URL discovery from Wayback Machine
- **Nuclei** - Fast vulnerability scanner with 4000+ templates
- **Nikto** - Web server vulnerability scanner with comprehensive checks
- **SQLMap** - Advanced automatic SQL injection testing with tamper scripts
- **WPScan** - WordPress security scanner with vulnerability database
- **Arjun** - HTTP parameter discovery with intelligent fuzzing
- **ParamSpider** - Parameter mining from web archives
- **X8** - Hidden parameter discovery with advanced techniques
- **Jaeles** - Advanced vulnerability scanning with custom signatures
- **Dalfox** - Advanced XSS vulnerability scanning with DOM analysis
- **Wafw00f** - Web application firewall fingerprinting
- **TestSSL** - SSL/TLS configuration testing and vulnerability assessment
- **SSLScan** - SSL/TLS cipher suite enumeration
- **SSLyze** - Fast and comprehensive SSL/TLS configuration analyzer
- **Anew** - Append new lines to files for efficient data processing
- **QSReplace** - Query string parameter replacement for systematic testing
- **Uro** - URL filtering and deduplication for efficient testing
- **Whatweb** - Web technology identification with fingerprinting
- **JWT-Tool** - JSON Web Token testing with algorithm confusion
- **GraphQL-Voyager** - GraphQL schema exploration and introspection testing
- **Burp Suite Extensions** - Custom extensions for advanced web testing
- **ZAP Proxy** - OWASP ZAP integration for automated security scanning
- **Wfuzz** - Web application fuzzer with advanced payload generation
- **Commix** - Command injection exploitation tool with automated detection
- **NoSQLMap** - NoSQL injection testing for MongoDB, CouchDB, etc.
- **Tplmap** - Server-side template injection exploitation tool

**🌐 Advanced Browser Agent:**

- **Headless Chrome Automation** - Full Chrome browser automation with Selenium
- **Screenshot Capture** - Automated screenshot generation for visual inspection
- **DOM Analysis** - Deep DOM tree analysis and JavaScript execution monitoring
- **Network Traffic Monitoring** - Real-time network request/response logging
- **Security Header Analysis** - Comprehensive security header validation
- **Form Detection & Analysis** - Automatic form discovery and input field analysis
- **JavaScript Execution** - Dynamic content analysis with full JavaScript support
- **Proxy Integration** - Seamless integration with Burp Suite and other proxies
- **Multi-page Crawling** - Intelligent web application spidering and mapping
- **Performance Metrics** - Page load times, resource usage, and optimization insights

</details>

<details>
<summary><b>🔐 Authentication & Password Security</b></summary>

- **Hydra** - Network login cracker supporting 50+ protocols
- **John the Ripper** - Advanced password hash cracking with custom rules
- **Hashcat** - World's fastest password recovery tool with GPU acceleration
- **Medusa** - Speedy, parallel, modular login brute-forcer
- **Patator** - Multi-purpose brute-forcer with advanced modules
- **NetExec** - Swiss army knife for pentesting networks
- **SMBMap** - SMB share enumeration and exploitation tool
- **Evil-WinRM** - Windows Remote Management shell with PowerShell integration
- **HashID** - Advanced hash algorithm identifier with confidence scoring
- **CrackStation** - Online hash lookup integration
- **Ophcrack** - Windows password cracker using rainbow tables

</details>

<details>
<summary><b>🔬 Binary Analysis & Reverse Engineering</b></summary>

- **GDB** - GNU Debugger with Python scripting and exploit development support
- **GDB-PEDA** - Python Exploit Development Assistance for GDB
- **GDB-GEF** - GDB Enhanced Features for exploit development
- **Radare2** - Advanced reverse engineering framework with comprehensive analysis
- **Ghidra** - NSA's software reverse engineering suite with headless analysis
- **IDA Free** - Interactive disassembler with advanced analysis capabilities
- **Binary Ninja** - Commercial reverse engineering platform
- **Binwalk** - Firmware analysis and extraction tool with recursive extraction
- **ROPgadget** - ROP/JOP gadget finder with advanced search capabilities
- **Ropper** - ROP gadget finder and exploit development tool
- **One-Gadget** - Find one-shot RCE gadgets in libc
- **Checksec** - Binary security property checker with comprehensive analysis
- **Strings** - Extract printable strings from binaries with filtering
- **Objdump** - Display object file information with Intel syntax
- **Readelf** - ELF file analyzer with detailed header information
- **XXD** - Hex dump utility with advanced formatting
- **Hexdump** - Hex viewer and editor with customizable output
- **Pwntools** - CTF framework and exploit development library
- **Angr** - Binary analysis platform with symbolic execution
- **Libc-Database** - Libc identification and offset lookup tool
- **Pwninit** - Automate binary exploitation setup
- **Volatility** - Advanced memory forensics framework
- **MSFVenom** - Metasploit payload generator with advanced encoding
- **UPX** - Executable packer/unpacker for binary analysis

</details>

<details>
<summary><b>☁️ Cloud & Container Security</b></summary>

- **Prowler** - AWS/Azure/GCP security assessment with compliance checks
- **Scout Suite** - Multi-cloud security auditing for AWS, Azure, GCP, Alibaba Cloud
- **CloudMapper** - AWS network visualization and security analysis
- **Pacu** - AWS exploitation framework with comprehensive modules
- **Trivy** - Comprehensive vulnerability scanner for containers and IaC
- **Clair** - Container vulnerability analysis with detailed CVE reporting
- **Kube-Hunter** - Kubernetes penetration testing with active/passive modes
- **Kube-Bench** - CIS Kubernetes benchmark checker with remediation
- **Docker Bench Security** - Docker security assessment following CIS benchmarks
- **Falco** - Runtime security monitoring for containers and Kubernetes
- **Checkov** - Infrastructure as code security scanning
- **Terrascan** - Infrastructure security scanner with policy-as-code
- **CloudSploit** - Cloud security scanning and monitoring
- **AWS CLI** - Amazon Web Services command line with security operations
- **Azure CLI** - Microsoft Azure command line with security assessment
- **GCloud** - Google Cloud Platform command line with security tools
- **Kubectl** - Kubernetes command line with security context analysis
- **Helm** - Kubernetes package manager with security scanning
- **Istio** - Service mesh security analysis and configuration assessment
- **OPA** - Policy engine for cloud-native security and compliance

</details>

<details>
<summary><b>🏆 CTF & Forensics Tools</b></summary>

- **Volatility** - Advanced memory forensics framework with comprehensive plugins
- **Volatility3** - Next-generation memory forensics with enhanced analysis
- **Foremost** - File carving and data recovery with signature-based detection
- **PhotoRec** - File recovery software with advanced carving capabilities
- **TestDisk** - Disk partition recovery and repair tool
- **Steghide** - Steganography detection and extraction with password support
- **Stegsolve** - Steganography analysis tool with visual inspection
- **Zsteg** - PNG/BMP steganography detection tool
- **Outguess** - Universal steganographic tool for JPEG images
- **ExifTool** - Metadata reader/writer for various file formats
- **Binwalk** - Firmware analysis and reverse engineering with extraction
- **Scalpel** - File carving tool with configurable headers and footers
- **Bulk Extractor** - Digital forensics tool for extracting features
- **Autopsy** - Digital forensics platform with timeline analysis
- **Sleuth Kit** - Collection of command-line digital forensics tools

**Cryptography & Hash Analysis:**

- **John the Ripper** - Password cracker with custom rules and advanced modes
- **Hashcat** - GPU-accelerated password recovery with 300+ hash types
- **HashID** - Hash type identification with confidence scoring
- **CyberChef** - Web-based analysis toolkit for encoding and encryption
- **Cipher-Identifier** - Automatic cipher type detection and analysis
- **Frequency-Analysis** - Statistical cryptanalysis for substitution ciphers
- **RSATool** - RSA key analysis and common attack implementations
- **FactorDB** - Integer factorization database for cryptographic challenges

</details>

<details>
<summary><b>🔥 Bug Bounty & OSINT Arsenal</b></summary>

- **Amass** - Advanced subdomain enumeration and OSINT gathering
- **Subfinder** - Fast passive subdomain discovery with API integration
- **Hakrawler** - Fast web endpoint discovery and crawling
- **HTTPx** - Fast and multi-purpose HTTP toolkit with technology detection
- **ParamSpider** - Mining parameters from web archives
- **Aquatone** - Visual inspection of websites across hosts
- **Subjack** - Subdomain takeover vulnerability checker
- **DNSEnum** - DNS enumeration script with zone transfer capabilities
- **Fierce** - Domain scanner for locating targets with DNS analysis
- **Sherlock** - Username investigation across 400+ social networks
- **Social-Analyzer** - Social media analysis and OSINT gathering
- **Recon-ng** - Web reconnaissance framework with modular architecture
- **Maltego** - Link analysis and data mining for OSINT investigations
- **SpiderFoot** - OSINT automation with 200+ modules
- **Shodan** - Internet-connected device search with advanced filtering
- **Censys** - Internet asset discovery with certificate analysis
- **Have I Been Pwned** - Breach data analysis and credential exposure
- **Pipl** - People search engine integration for identity investigation
- **TruffleHog** - Git repository secret scanning with entropy analysis

</details>

<details>
<summary><b>📡 WiFi Penetration Testing</b></summary>

- Aircrack-ng Suite:
- Aircrack-ng - WPA/WPA2 PSK cracking from captured handshakes using dictionary attacks
- Airmon-ng - Enable/disable monitor mode and kill interfering processes
- Airodump-ng - Passive 802.11 packet capture for AP discovery and WPA handshake collection
- Aireplay-ng - Packet injection for deauthentication, fake authentication, and ARP replay attacks
- Airbase-ng - Rogue/soft access point creation for Evil Twin and client capture attacks
- Airdecap-ng - Decrypt WEP/WPA/WPA2 encrypted pcap capture files

*Modern WiFi Tools:*

- hcxdumptool - Clientless PMKID capture and WPA/WPA2 handshake collection (v7.0.0+)
- hcxpcapngtool - Convert hcxdumptool pcapng output to hashcat -m 22000 format
- EAPHammer - WPA-Enterprise Evil Twin for harvesting 802.1X EAP credentials
- Wifite2 - Automated WiFi auditing with PMKID, handshake, and WPS attack support
- Bettercap - WiFi recon, deauthentication, and Evil Twin via Bettercap wifi module
- mdk4 - 802.11 protocol stress testing and WIDS/WIPS evasion validation

</details>

---

## Legal

This software is intended solely for **authorized security testing, research, and educational purposes**.

You may only use this software on systems, networks, or applications for which you have **explicit written permission** from the owner. Unauthorized use is strictly prohibited and may violate local, national, or international laws.

The authors assume no liability for unauthorized or illegal use.
