import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from fastmcp import FastMCP, Context
from mcp_tools.gateway import register_gateway_tools
from mcp_core.tool_profiles import (
    TOOL_PROFILES,
    DEFAULT_PROFILE,
    FULL_PROFILE,
    resolve_profile_dependencies,
)

try:
    from fastmcp.server.providers.skills import SkillsDirectoryProvider
except ImportError:
    SkillsDirectoryProvider = None

try:
    from fastmcp.server.transforms.search import BM25SearchTransform
except ImportError:
    BM25SearchTransform = None

# ---------------------------------------------------------------------------
# In-memory scan result cache — populated by run_security_tool
# ---------------------------------------------------------------------------
_scan_cache: Dict[str, Dict] = {}
_server_start_time = time.time()


def _register_skills(mcp: FastMCP, logger) -> None:
    """Mount the local skills/ directory as MCP resources if it exists."""
    if SkillsDirectoryProvider is None:
        logger.warning("fastmcp SkillsDirectoryProvider not available; skipping skills registration")
        return
    skills_dir = Path(__file__).parent.parent / "skills"
    if not skills_dir.exists():
        return
    mcp.add_provider(SkillsDirectoryProvider(roots=skills_dir))
    logger.info(f"🤖 Skills initialized")

def setup_mcp_server(hexstrike_client, logger, compact: bool = False, profiles: Optional[list] = None) -> FastMCP:
    """
    Set up the MCP server with all enhanced tool functions (Phase 2 — Flask still present).

    Args:
        hexstrike_client: Initialized HexStrikeClient
        logger: Logger instance
        compact: If True, register only classify_task and run_tool gateway tools
        profiles: Optional list of tool profiles to load

    Returns:
        Configured FastMCP instance
    """
    transforms = [BM25SearchTransform()] if BM25SearchTransform else []
    mcp = FastMCP("hexstrike-ai-mcp", transforms=transforms)

    _register_skills(mcp, logger)

    if compact:
        register_gateway_tools(mcp, hexstrike_client)
        logger.info("Compact mode: only gateway tools registered (classify_task, run_tool)")
        return mcp

    if profiles:
        if "default" in profiles:
            selected_profiles = DEFAULT_PROFILE
        elif "full" in profiles:
            selected_profiles = FULL_PROFILE
        else:
            selected_profiles = profiles
    else:
        selected_profiles = DEFAULT_PROFILE

    selected_profiles = resolve_profile_dependencies(selected_profiles)

    registered = set()
    for profile in selected_profiles:
        for reg_func in TOOL_PROFILES.get(profile, []):
            if reg_func not in registered:
                reg_func(mcp, hexstrike_client, logger)
                registered.add(reg_func)

    return mcp

def setup_mcp_server_standalone(logger=None) -> FastMCP:
    """
    Set up the MCP server in standalone mode (Phase 3).

    No Flask dependency, no HTTP round-trips.
    All tools route directly through *_direct.py modules.

    Args:
        logger: Optional logger instance

    Returns:
        Configured FastMCP instance ready for mcp.run(transport="http")
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    transforms = [BM25SearchTransform()] if BM25SearchTransform else []
    mcp = FastMCP("hexstrike-ai-mcp", transforms=transforms)

    _register_skills(mcp, logger)

    # Import all direct execution modules
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

    DIRECT_TOOLS = {
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

    # ========================================================================
    # Main tool — run any security tool by name
    # ========================================================================

    @mcp.tool(description="Execute any HexStrike security tool by name with JSON parameters")
    async def run_security_tool(
        ctx: Context,
        tool_name: str,
        parameters: str,
    ) -> Dict[str, Any]:
        """
        Execute any security tool from the HexStrike arsenal.

        Args:
            tool_name:  Tool name (e.g. 'nmap', 'sqlmap', 'sherlock')
            parameters: JSON string of parameters (e.g. '{"target": "example.com"}')

        Returns:
            Tool execution results
        """
        await ctx.info(f"🔍 Executing {tool_name}")

        try:
            params = json.loads(parameters) if isinstance(parameters, str) else parameters
        except json.JSONDecodeError as e:
            await ctx.error(f"❌ Invalid JSON parameters: {e}")
            return {"success": False, "error": f"Invalid JSON: {e}"}

        route = DIRECT_TOOLS.get(tool_name.lower())
        if not route:
            await ctx.error(f"❌ Unknown tool: {tool_name}")
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        exec_func, tool_key = route

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: exec_func(tool_key, params)
        )

        if result.get("success"):
            await ctx.info(f"✅ {tool_name} completed")
            # Cache result for Resource access
            target = params.get("target") or params.get("domain") or params.get("interface", "")
            if target:
                cache_key = f"{tool_name}:{target}"
                _scan_cache[cache_key] = {
                    "tool":      tool_name,
                    "target":    target,
                    "result":    result,
                    "timestamp": time.time(),
                }
        else:
            await ctx.error(f"❌ {tool_name} failed: {result.get('error', 'unknown')}")

        return result

    logger.info(f"🚀 Phase 3: {len(DIRECT_TOOLS)} tools registered — no Flask")

    # ========================================================================
    # Resources MCP — health + scan results
    # ========================================================================

    @mcp.resource("health://server")
    async def server_health() -> str:
        """Server health and runtime statistics."""
        uptime = int(time.time() - _server_start_time)
        return json.dumps({
            "status":         "healthy",
            "server":         "hexstrike-ai-mcp",
            "fastmcp":        "3.1.1",
            "uptime_seconds": uptime,
            "tools_count":    len(DIRECT_TOOLS),
            "cached_scans":   len(_scan_cache),
        }, indent=2)

    @mcp.resource("scan://{target}/latest")
    async def scan_latest(target: str) -> str:
        """Most recent scan result for a given target across all tools."""
        matches = [
            v for k, v in _scan_cache.items()
            if v.get("target") == target
        ]
        if not matches:
            return json.dumps({
                "target": target,
                "status": "no_results",
                "message": f"No scan results cached for {target}",
            }, indent=2)

        latest = max(matches, key=lambda x: x["timestamp"])
        return json.dumps({
            "target":    target,
            "tool":      latest["tool"],
            "timestamp": latest["timestamp"],
            "result":    latest["result"],
        }, indent=2)

    @mcp.resource("scan://{target}/{tool_name}")
    async def scan_result(target: str, tool_name: str) -> str:
        """Cached result for a specific tool + target combination."""
        cache_key = f"{tool_name}:{target}"
        entry = _scan_cache.get(cache_key)
        if not entry:
            return json.dumps({
                "target":    target,
                "tool":      tool_name,
                "status":    "no_results",
                "message":   f"No cached result for {tool_name} on {target}",
            }, indent=2)

        return json.dumps({
            "target":    target,
            "tool":      tool_name,
            "timestamp": entry["timestamp"],
            "result":    entry["result"],
        }, indent=2)

    @mcp.resource("scan://cache/list")
    async def scan_cache_list() -> str:
        """List all cached scan results with timestamps."""
        entries = [
            {
                "key":       k,
                "tool":      v["tool"],
                "target":    v["target"],
                "timestamp": v["timestamp"],
                "success":   v["result"].get("success", False),
            }
            for k, v in _scan_cache.items()
        ]
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return json.dumps({"count": len(entries), "scans": entries}, indent=2)

    logger.info("📦 Resources MCP registered: health://server, scan://{target}/{tool}")

    # ========================================================================
    # Workflow Prompts — native MCP prompts for multi-tool attack chains
    # ========================================================================

    from mcp_core.prompts import register_prompts
    register_prompts(mcp)
    logger.info("🎯 Workflow prompts registered: bug_bounty_recon, wifi_attack_chain, ctf_web_challenge, smb_lateral_movement, cloud_security_audit")

    return mcp
