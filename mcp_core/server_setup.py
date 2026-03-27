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
    Set up the MCP server with all enhanced tool functions

    Args:
        hexstrike_client: Initialized HexStrikeClient
        logger: Logger instance for logging
        compact: If True, register only classify_task and run_tool gateway tools
        profiles: Optional list of tool profiles to load (e.g., ["core_network", "web_app"])

    Returns:
        Configured FastMCP instance
    """
    transforms = [BM25SearchTransform()] if BM25SearchTransform else []
    mcp = FastMCP("hexstrike-ai-mcp", transforms=transforms)

    _register_skills(mcp, logger)

    if compact:
        # Register gateway tools for task classification and tool execution
        register_gateway_tools(mcp, hexstrike_client)

        logger.info("Compact mode: only gateway tools registered (classify_task, run_tool)")
        return mcp

    # Determine which profiles to load
    if profiles:
        if "default" in profiles:
            selected_profiles = DEFAULT_PROFILE
        elif "full" in profiles:
            selected_profiles = FULL_PROFILE
        else:
            selected_profiles = profiles
    else:
        selected_profiles = DEFAULT_PROFILE

    # Register tools for each selected profile
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
    
    Single generic tool interface for all security tools.
    No Flask dependency, no HTTP round-trips.
    
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
    
    # Tool registry mapping tool names to (exec_func, tool_key)
    DIRECT_TOOLS = {
        # wifi tools
        "airmon_ng": (wifi_exec, "airmon_ng"),
        "airodump_ng": (wifi_exec, "airodump_ng"), 
        "aireplay_ng": (wifi_exec, "aireplay_ng"),
        "aircrack_ng": (wifi_exec, "aircrack_ng"),
        "hcxdumptool": (wifi_exec, "hcxdumptool"),
        "wifite": (wifi_exec, "wifite2"),
        "wifite2": (wifi_exec, "wifite2"),
        # recon tools
        "amass": (recon_exec, "amass"),
        "subfinder": (recon_exec, "subfinder"),
        "autorecon": (recon_exec, "autorecon"),
        "theharvester": (recon_exec, "theharvester"),
        "dnsenum": (recon_exec, "dnsenum"),
        "fierce": (recon_exec, "fierce"),
        "whois": (recon_exec, "whois"),
        # net_scan tools
        "nmap": (net_scan_exec, "nmap"),
        "nmap_advanced": (net_scan_exec, "nmap-advanced"),
        "masscan": (net_scan_exec, "masscan"),
        "rustscan": (net_scan_exec, "rustscan"),
        "arp_scan": (net_scan_exec, "arp-scan"),
        # web_scan tools
        "nikto": (web_scan_exec, "nikto"),
        "sqlmap": (web_scan_exec, "sqlmap"),
        "wpscan": (web_scan_exec, "wpscan"),
        "dalfox": (web_scan_exec, "dalfox"),
        "jaeles": (web_scan_exec, "jaeles"),
        "xsser": (web_scan_exec, "xsser"),
        "zap": (web_scan_exec, "zap"),
        # web_fuzz tools
        "gobuster": (web_fuzz_exec, "gobuster"),
        "ffuf": (web_fuzz_exec, "ffuf"),
        "feroxbuster": (web_fuzz_exec, "feroxbuster"),
        "dirsearch": (web_fuzz_exec, "dirsearch"),
        "dirb": (web_fuzz_exec, "dirb"),
        "wfuzz": (web_fuzz_exec, "wfuzz"),
        "dotdotpwn": (web_fuzz_exec, "dotdotpwn"),
        # password_cracking tools
        "hydra": (pwdcrack_exec, "hydra"),
        "hashcat": (pwdcrack_exec, "hashcat"),
        "john": (pwdcrack_exec, "john"),
        "medusa": (pwdcrack_exec, "medusa"),
        "patator": (pwdcrack_exec, "patator"),
        "hashid": (pwdcrack_exec, "hashid"),
        "ophcrack": (pwdcrack_exec, "ophcrack"),
        # smb_enum tools
        "enum4linux": (smb_enum_exec, "enum4linux"),
        "netexec": (smb_enum_exec, "netexec"),
        "rpcclient": (smb_enum_exec, "rpcclient"),
        "smbmap": (smb_enum_exec, "smbmap"),
        "nbtscan": (smb_enum_exec, "nbtscan"),
        # exploit tools
        "metasploit": (exploit_exec, "metasploit"),
        "msfvenom": (exploit_exec, "msfvenom"),
        "searchsploit": (exploit_exec, "exploit_db"),
        "exploit_db": (exploit_exec, "exploit_db"),
        # web_recon tools
        "katana": (web_recon_exec, "katana"),
        "hakrawler": (web_recon_exec, "hakrawler"),
        "gau": (web_recon_exec, "gau"),
        "waybackurls": (web_recon_exec, "waybackurls"),
        "httpx": (web_recon_exec, "httpx"),
        "wafw00f": (web_recon_exec, "wafw00f"),
        "arjun": (web_recon_exec, "arjun"),
        "paramspider": (web_recon_exec, "paramspider"),
        "x8": (web_recon_exec, "x8"),
        # security tools
        "prowler": (security_exec, "prowler"),
        "trivy": (security_exec, "trivy"),
        "kube_hunter": (security_exec, "kube-hunter"),
        "kube_bench": (security_exec, "kube-bench"),
        "checkov": (security_exec, "checkov"),
        "terrascan": (security_exec, "terrascan"),
        # misc tools
        "ropgadget": (misc_exec, "ropgadget"),
        "ropper": (misc_exec, "ropper"),
        "one_gadget": (misc_exec, "one_gadget"),
        "volatility": (misc_exec, "volatility"),
        "volatility3": (misc_exec, "volatility3"),
        "gdb": (misc_exec, "gdb"),
        "radare2": (misc_exec, "radare2"),
        "strings": (misc_exec, "strings"),
        "objdump": (misc_exec, "objdump"),
        "checksec": (misc_exec, "checksec"),
        "binwalk": (misc_exec, "binwalk"),
        "ghidra": (misc_exec, "ghidra"),
        "angr": (misc_exec, "angr"),
        "xxd": (misc_exec, "xxd"),
        "mysql": (misc_exec, "mysql"),
        "sqlite": (misc_exec, "sqlite"),
        "exiftool": (misc_exec, "exiftool"),
        "foremost": (misc_exec, "foremost"),
        "steghide": (misc_exec, "steghide"),
        "hashpump": (misc_exec, "hashpump"),
        "anew": (misc_exec, "anew"),
        "uro": (misc_exec, "uro"),
        "nuclei": (misc_exec, "nuclei"),
        "responder": (misc_exec, "responder"),
        # osint tools
        "sherlock": (osint_exec, "sherlock"),
        "spiderfoot": (osint_exec, "spiderfoot"),
        "sublist3r": (osint_exec, "sublist3r"),
        "parsero": (osint_exec, "parsero"),
    }
    
    # Register a single generic tool that can execute any security tool
    @mcp.tool(description="Execute any HexStrike security tool by name with parameters")
    async def run_security_tool(
        tool_name: str, 
        parameters: str,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Execute any security tool from the HexStrike arsenal.
        
        Args:
            tool_name: Name of the tool to execute (e.g., 'nmap', 'sqlmap', 'sherlock')
            parameters: JSON string of tool parameters (e.g., '{"target": "example.com"}')
            
        Returns:
            Tool execution results
        """
        if ctx:
            await ctx.info(f"🔍 Executing {tool_name}")
        
        # Parse parameters
        import json
        try:
            params = json.loads(parameters) if isinstance(parameters, str) else parameters
        except json.JSONDecodeError:
            error_msg = "Invalid JSON parameters"
            if ctx:
                await ctx.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        # Find the tool
        route = DIRECT_TOOLS.get(tool_name.lower())
        if not route:
            error_msg = f"Unknown tool: {tool_name}"
            if ctx:
                await ctx.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        exec_func, tool_key = route
        
        # Execute the tool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: exec_func(tool_key, params)
        )
        
        if ctx:
            if result.get("success"):
                await ctx.info(f"✅ {tool_name} completed")
            else:
                await ctx.error(f"❌ {tool_name} failed: {result.get('error', 'unknown')}")
        
        return result
    
    logger.info(f"🚀 Phase 3: Registered generic tool interface for {len(DIRECT_TOOLS)} security tools")
    return mcp
