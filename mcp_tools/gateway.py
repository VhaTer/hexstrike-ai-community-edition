# mcp_tools/gateway.py

from typing import Dict, Any
import json
import asyncio

def register_gateway_tools(mcp, hexstrike_client):
    @mcp.tool()
    async def classify_task(description: str) -> Dict[str, Any]:
        """
        Classify a security task and return recommended tools.
        Call this FIRST before running security tools to discover which ones are relevant.

        Args:
            description: What you want to do (e.g., "scan for open ports", "test for SQL injection")

        Returns:
            Task category, recommended tools with parameters, and usage instructions
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/intelligence/classify-task", {"description": description})
        )
        if result.get("success"):
            result["usage"] = "Use run_tool with a tool name and params from the recommended list"
        return result

    @mcp.tool()
    async def run_tool(
        tool_name: str,
        params: str,
    ) -> Dict[str, Any]:
        """
        Execute any security tool by name with parameters.
        Use classify_task first to discover available tools.

        Args:
            tool_name: Tool name from classify_task results (e.g., "nmap", "hashid")
            params: JSON string of parameters (e.g., '{"target": "10.0.0.1"}')

        Returns:
            Tool execution results
        """
        try:
            parsed_params = json.loads(params) if isinstance(params, str) else params
        except json.JSONDecodeError as e:
            return {"error": f"Invalid params JSON: {e}", "success": False}

        # Route through *_direct.py modules — no Flask, no HTTP
        from mcp_core.wifi_direct import wifi_exec
        from mcp_core.recon_direct import recon_exec
        from mcp_core.net_scan_direct import net_scan_exec
        from mcp_core.web_scan_direct import web_scan_exec
        from mcp_core.web_fuzz_direct import web_fuzz_exec
        from mcp_core.password_cracking_direct import pwdcrack_exec
        from mcp_core.smb_enum_direct import smb_enum_exec
        from mcp_core.exploit_framework_direct import exploit_exec
        from mcp_core.web_recon_direct import web_recon_exec
        from mcp_core.security_direct import security_exec
        from mcp_core.misc_direct import misc_exec
        from mcp_core.osint_direct import osint_exec
        from mcp_core.active_directory_direct import ad_exec

        DIRECT_ROUTES = {
            # wifi
            "airmon_ng":    (wifi_exec, "airmon_ng"),
            "airodump_ng":  (wifi_exec, "airodump_ng"),
            "aireplay_ng":  (wifi_exec, "aireplay_ng"),
            "aircrack_ng":  (wifi_exec, "aircrack_ng"),
            "hcxdumptool":  (wifi_exec, "hcxdumptool"),
            "wifite":       (wifi_exec, "wifite2"),
            "wifite2":      (wifi_exec, "wifite2"),
            # recon
            "amass":        (recon_exec, "amass"),
            "subfinder":    (recon_exec, "subfinder"),
            "autorecon":    (recon_exec, "autorecon"),
            "theharvester": (recon_exec, "theharvester"),
            "dnsenum":      (recon_exec, "dnsenum"),
            "fierce":       (recon_exec, "fierce"),
            "whois":        (recon_exec, "whois"),
            # net_scan
            "nmap":         (net_scan_exec, "nmap"),
            "nmap_advanced":(net_scan_exec, "nmap-advanced"),
            "masscan":      (net_scan_exec, "masscan"),
            "rustscan":     (net_scan_exec, "rustscan"),
            "arp_scan":     (net_scan_exec, "arp-scan"),
            # web_scan
            "nikto":        (web_scan_exec, "nikto"),
            "sqlmap":       (web_scan_exec, "sqlmap"),
            "wpscan":       (web_scan_exec, "wpscan"),
            "dalfox":       (web_scan_exec, "dalfox"),
            "jaeles":       (web_scan_exec, "jaeles"),
            "xsser":        (web_scan_exec, "xsser"),
            "zap":          (web_scan_exec, "zap"),
            # web_fuzz
            "gobuster":     (web_fuzz_exec, "gobuster"),
            "ffuf":         (web_fuzz_exec, "ffuf"),
            "feroxbuster":  (web_fuzz_exec, "feroxbuster"),
            "dirsearch":    (web_fuzz_exec, "dirsearch"),
            "dirb":         (web_fuzz_exec, "dirb"),
            "wfuzz":        (web_fuzz_exec, "wfuzz"),
            "dotdotpwn":    (web_fuzz_exec, "dotdotpwn"),
            # password_cracking
            "hydra":        (pwdcrack_exec, "hydra"),
            "hashcat":      (pwdcrack_exec, "hashcat"),
            "john":         (pwdcrack_exec, "john"),
            "medusa":       (pwdcrack_exec, "medusa"),
            "patator":      (pwdcrack_exec, "patator"),
            "hashid":       (pwdcrack_exec, "hashid"),
            "ophcrack":     (pwdcrack_exec, "ophcrack"),
            # smb_enum
            "enum4linux":   (smb_enum_exec, "enum4linux"),
            "netexec":      (smb_enum_exec, "netexec"),
            "rpcclient":    (smb_enum_exec, "rpcclient"),
            "smbmap":       (smb_enum_exec, "smbmap"),
            "nbtscan":      (smb_enum_exec, "nbtscan"),
            # exploit
            "metasploit":   (exploit_exec, "metasploit"),
            "msfvenom":     (exploit_exec, "msfvenom"),
            "searchsploit": (exploit_exec, "exploit_db"),
            "exploit_db":   (exploit_exec, "exploit_db"),
            # web_recon
            "katana":       (web_recon_exec, "katana"),
            "hakrawler":    (web_recon_exec, "hakrawler"),
            "gau":          (web_recon_exec, "gau"),
            "waybackurls":  (web_recon_exec, "waybackurls"),
            "httpx":        (web_recon_exec, "httpx"),
            "wafw00f":      (web_recon_exec, "wafw00f"),
            "arjun":        (web_recon_exec, "arjun"),
            "paramspider":  (web_recon_exec, "paramspider"),
            "x8":           (web_recon_exec, "x8"),
            # security
            "prowler":      (security_exec, "prowler"),
            "trivy":        (security_exec, "trivy"),
            "kube_hunter":  (security_exec, "kube-hunter"),
            "kube_bench":   (security_exec, "kube-bench"),
            "checkov":      (security_exec, "checkov"),
            "terrascan":    (security_exec, "terrascan"),
            # misc
            "ropgadget":    (misc_exec, "ropgadget"),
            "ropper":       (misc_exec, "ropper"),
            "one_gadget":   (misc_exec, "one_gadget"),
            "volatility":   (misc_exec, "volatility"),
            "volatility3":  (misc_exec, "volatility3"),
            "gdb":          (misc_exec, "gdb"),
            "radare2":      (misc_exec, "radare2"),
            "strings":      (misc_exec, "strings"),
            "objdump":      (misc_exec, "objdump"),
            "checksec":     (misc_exec, "checksec"),
            "binwalk":      (misc_exec, "binwalk"),
            "ghidra":       (misc_exec, "ghidra"),
            "angr":         (misc_exec, "angr"),
            "xxd":          (misc_exec, "xxd"),
            "mysql":        (misc_exec, "mysql"),
            "sqlite":       (misc_exec, "sqlite"),
            "exiftool":     (misc_exec, "exiftool"),
            "foremost":     (misc_exec, "foremost"),
            "steghide":     (misc_exec, "steghide"),
            "hashpump":     (misc_exec, "hashpump"),
            "anew":         (misc_exec, "anew"),
            "uro":          (misc_exec, "uro"),
            "nuclei":       (misc_exec, "nuclei"),
            "responder":    (misc_exec, "responder"),
            # osint
            "sherlock":            (osint_exec, "sherlock"),
            "spiderfoot":          (osint_exec, "spiderfoot"),
            "sublist3r":           (osint_exec, "sublist3r"),
            "parsero":             (osint_exec, "parsero"),
            # active_directory
            "impacket":            (ad_exec, "impacket"),
            "ldapdomaindump":      (ad_exec, "ldapdomaindump"),
            "adidnsdump":          (ad_exec, "adidnsdump"),
            "certipy":             (ad_exec, "certipy_ad"),
            "certipy_ad":          (ad_exec, "certipy_ad"),
            "mitm6":               (ad_exec, "mitm6"),
            "pywerview":           (ad_exec, "pywerview"),
            "bloodhound":          (ad_exec, "bloodhound"),
            "bloodhound_python":   (ad_exec, "bloodhound"),
        }

        route = DIRECT_ROUTES.get(tool_name.lower())
        if route:
            exec_fn, tool_key = route
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: exec_fn(tool_key, parsed_params)
            )

        # Fallback — tool not yet migrated, use Flask
        from tool_registry import get_tool
        tool_def = get_tool(tool_name)
        if not tool_def:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

        for pname, spec in tool_def["params"].items():
            if spec.get("required") and pname not in parsed_params:
                return {"error": f"Missing required param: {pname}", "success": False}

        for k, v in tool_def.get("optional", {}).items():
            if k not in parsed_params:
                parsed_params[k] = v

        endpoint = tool_def["endpoint"].lstrip("/")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post(endpoint, parsed_params)
        )
