# mcp_tools/net_scan/nmap.py

from typing import Dict, Any
from fastmcp import Context

def register_nmap(mcp, hexstrike_client, logger=None, HexStrikeColors=None):

    @mcp.tool()
    async def nmap_scan(
        ctx: Context,
        target: str,
        scan_type: str = "-sV",
        ports: str = "",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Execute an Nmap scan against a target for service and version detection.

        Workflow position: use after rustscan/masscan identifies open ports,
        or standalone for a complete scan on a single target.

        Parameters:
        - target: IP address, hostname, or CIDR range (e.g. 192.168.1.1, 10.0.0.0/24)
        - scan_type: Nmap scan flags (default: -sV for version detection)
            common values:
            '-sV'        — service/version detection
            '-sC'        — default NSE scripts
            '-sV -sC'    — version + scripts (recommended)
            '-sS'        — SYN stealth scan (requires root)
            '-sU'        — UDP scan
            '-A'         — aggressive (OS + version + scripts + traceroute)
        - ports: comma-separated ports or ranges (e.g. '22,80,443' or '1-1000')
                 omit to scan top 1000 ports
        - additional_args: extra nmap flags (e.g. '-T4 -O --open')

        Prerequisites: none — can be run standalone.
        For efficiency: run rustscan_fast_scan first to discover open ports,
        then pass those ports here for deep service enumeration.

        Output: service names, versions, states, and script output per port.

        Recommended sequence:
            1. rustscan_fast_scan(target='192.168.1.1', ports='1-65535')
            2. nmap_scan(target='192.168.1.1', scan_type='-sV -sC',
                         ports='<open ports from rustscan>')
        """
        data: Dict[str, Any] = {
            "target": target,
            "scan_type": scan_type,
            "ports": ports,
            "additional_args": additional_args,
            "use_recovery": True
        }
        await ctx.info(f"🔍 Starting nmap scan: {target}")
        result = hexstrike_client.safe_post("api/tools/nmap", data)
        if result.get("success"):
            await ctx.info(f"✅ nmap scan completed for {target}")
            if result.get("recovery_info", {}).get("recovery_applied"):
                attempts = result["recovery_info"].get("attempts_made", 1)
                await ctx.info(f"⚠️ Recovery applied: {attempts} attempt(s)")
        else:
            await ctx.error(f"❌ nmap scan failed for {target}")
            if result.get("human_escalation"):
                await ctx.error("🚨 HUMAN ESCALATION REQUIRED")
        return result

    @mcp.tool()
    async def nmap_advanced_scan(
        ctx: Context,
        target: str,
        scan_type: str = "-sS",
        ports: str = "",
        timing: str = "T4",
        nse_scripts: str = "",
        os_detection: bool = False,
        version_detection: bool = False,
        aggressive: bool = False,
        stealth: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Execute advanced Nmap scans with custom NSE scripts and optimized timing.

        Workflow position: use after nmap_scan identifies services,
        for targeted deep enumeration of specific ports/services.

        Parameters:
        - target: IP address or hostname
        - scan_type: base scan type ('-sS' SYN stealth, '-sT' TCP connect, '-sU' UDP)
        - ports: specific ports to target (e.g. '445' for SMB, '80,443' for web)
        - timing: T0 (paranoid) to T5 (insane) — default T4 (aggressive)
        - nse_scripts: NSE script names or categories
            examples:
            'smb-vuln-*,smb-enum-shares'     — SMB vulnerabilities
            'http-title,http-headers'         — HTTP enumeration
            'ftp-anon,ftp-bounce'             — FTP checks
            'ssh-hostkey,ssh-auth-methods'    — SSH enumeration
            'vuln'                            — all vuln scripts
        - os_detection: enable OS fingerprinting (requires root)
        - version_detection: enable service version detection
        - aggressive: enable -A (OS + version + scripts + traceroute)
        - stealth: reduce scan noise (slower timing, fragmented packets)
        - additional_args: extra nmap flags

        Prerequisites: best used with known open ports from nmap_scan or rustscan.

        Common targeted sequences:
            SMB: nmap_advanced_scan(target='x.x.x.x', ports='445',
                     nse_scripts='smb-vuln-*,smb-enum-shares')
            Web: nmap_advanced_scan(target='x.x.x.x', ports='80,443',
                     nse_scripts='http-title,http-methods,http-headers')
        """
        data = {
            "target": target,
            "scan_type": scan_type,
            "ports": ports,
            "timing": timing,
            "nse_scripts": nse_scripts,
            "os_detection": os_detection,
            "version_detection": version_detection,
            "aggressive": aggressive,
            "stealth": stealth,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting advanced nmap: {target}")
        result = hexstrike_client.safe_post("api/tools/nmap-advanced", data)
        if result.get("success"):
            await ctx.info(f"✅ Advanced nmap completed for {target}")
        else:
            await ctx.error(f"❌ Advanced nmap failed for {target}")
        return result
