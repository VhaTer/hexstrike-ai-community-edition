import asyncio
import inspect
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_context
from mcp_tools.gateway import register_gateway_tools
from mcp_core.parameter_optimizer import ParameterOptimizer
from mcp_core.technology_detector import TechProfile, TechnologyDetector
from mcp_core.elicitation import confirm_destructive_action
from server_core.rate_limit_detector import RateLimitDetector
from tool_registry import get_tool
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
from server_core.advanced_cache import AdvancedCache as _AdvancedCache


class _ScanCache(_AdvancedCache):
    """Scan-specific cache: wraps AdvancedCache with adaptive TTL from execution_time."""
    _TTL_DEFAULT = 1800   # 30 min
    _TTL_MEDIUM  = 3600   # 60 min (exec > 10s)
    _TTL_LONG    = 5400   # 90 min (exec > 60s)

    def set(self, key: str, value: Any, execution_time: float = 0.0, ttl: Optional[int] = None) -> None:  # type: ignore[override]
        if ttl is None:
            if execution_time > 60:
                ttl = self._TTL_LONG
            elif execution_time > 10:
                ttl = self._TTL_MEDIUM
            else:
                ttl = self._TTL_DEFAULT
        super().set(key, value, ttl=ttl)

    def stats(self) -> Dict[str, Any]:
        return self.get_stats()


_scan_cache = _ScanCache(max_size=500, default_ttl=1800)
_server_start_time = time.time()
_optimizer    = ParameterOptimizer()
_detector     = TechnologyDetector()
_rate_limiter = RateLimitDetector()

# Tools requiring user confirmation before execution
_DESTRUCTIVE_TOOLS = {
    "aireplay_ng": ("Deauth/injection attack",      "Can disconnect all clients on the target AP"),
    "mdk4":        ("MDK4 wireless attack",         "Can cause denial of service on wireless networks"),
    "responder":   ("Responder LLMNR/NBT-NS poison", "Intercepts credentials on the local network"),
    "metasploit":  ("Metasploit exploit execution", "Will attempt to exploit the target system"),
    "mitm6":       ("MitM6 IPv6 attack",            "Poisons IPv6 DNS on the local network"),
}


def _build_destructive_confirmation(tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Mirror the safety rules already present in transitional wrappers.

    The standalone Phase 3 path must not be more permissive than the wrapper path,
    but it should keep the same safe exceptions:
    - aireplay-ng mode 9 injection test is allowed without confirmation
    - Responder analyze mode is passive and allowed without confirmation
    - Metasploit auxiliary scanner/gather modules are allowed without confirmation
    """
    tool = tool_name.lower()

    if tool == "aireplay_ng":
        try:
            attack_mode = int(params.get("attack_mode"))
        except (TypeError, ValueError):
            attack_mode = None

        if attack_mode == 9:
            return None

        interface = str(params.get("interface", "")).strip() or "unknown interface"
        bssid = str(params.get("bssid", "")).strip()
        client_mac = str(params.get("client_mac", "")).strip()
        target_info = f"BSSID: {bssid}" if bssid else "all visible networks"
        client_info = f" | Client: {client_mac}" if client_mac else " | all clients"
        warning = "This will disrupt active network connections." if attack_mode == 0 else ""
        mode_label = f"-{attack_mode}" if attack_mode is not None else "unknown mode"
        return {
            "action": f"aireplay-ng {mode_label} on {interface}",
            "detail": f"{target_info}{client_info}",
            "warning": warning,
        }

    if tool == "responder":
        if params.get("analyze", False):
            return None

        interface = str(params.get("interface", "eth0")).strip() or "eth0"
        duration = params.get("duration", 300)
        wpad = params.get("wpad", True)
        force_wpad_auth = params.get("force_wpad_auth", False)
        return {
            "action": f"Responder LLMNR/NBT-NS poisoning on {interface}",
            "detail": f"Duration: {duration}s | WPAD: {wpad} | Force WPAD auth: {force_wpad_auth}",
            "warning": "This actively poisons network traffic and may affect all hosts on the segment.",
        }

    if tool == "metasploit":
        module = str(params.get("module", "")).strip()
        if module.startswith("auxiliary/scanner/") or module.startswith("auxiliary/gather/"):
            return None

        options = params.get("options", {})
        if not isinstance(options, dict):
            options = {}
        rhosts = options.get("RHOSTS", options.get("rhosts", "unknown target"))
        return {
            "action": f"Metasploit: {module or 'unknown module'}",
            "detail": f"Target: {rhosts}",
            "warning": "Active exploitation - this may trigger IDS/IPS and alter target state.",
        }

    if tool == "mitm6":
        interface = str(params.get("interface", "")).strip() or "unknown interface"
        domain = str(params.get("domain", "")).strip()
        return {
            "action": f"mitm6 IPv6 DNS takeover on {interface}",
            "detail": f"Domain: {domain}" if domain else "All domains",
            "warning": "This poisons IPv6 DNS for all hosts on the network segment.",
        }

    destructive = _DESTRUCTIVE_TOOLS.get(tool)
    if destructive:
        action, warning = destructive
        target_hint = (
            params.get("target")
            or params.get("interface")
            or params.get("url")
            or params.get("domain")
            or ""
        )
        return {
            "action": f"{action}: {target_hint}" if target_hint else action,
            "detail": "",
            "warning": warning,
        }

    return None


def _detect_from_cache(target: str) -> Optional[TechProfile]:
    """
    Build a TechProfile from cached scan results for a given target.

    Looks for whatweb or httpx results in _scan_cache and passes
    their output text to TechnologyDetector.detect().
    Returns None if no usable cache entry is found.
    """
    # Tools whose output is useful for tech detection, in priority order
    PROBE_TOOLS = ("whatweb", "httpx", "nikto", "wpscan")

    content_parts = []
    headers: Dict[str, str] = {}

    for tool in PROBE_TOOLS:
        entry = _scan_cache.get(f"{tool}:{target}")
        if not entry:
            continue
        result = entry.get("result", {})
        # Most direct modules return output in result["output"] or result["data"]
        output = (
            result.get("output")
            or result.get("data")
            or result.get("stdout")
            or ""
        )
        if isinstance(output, str) and output.strip():
            content_parts.append(output)
        # httpx sometimes returns structured headers
        if tool == "httpx" and isinstance(result.get("headers"), dict):
            headers.update(result["headers"])

    if not content_parts and not headers:
        return None

    return _detector.detect(
        headers=headers,
        content="\n".join(content_parts),
    )

# ---------------------------------------------------------------------------
# Tool → Skill mapping for ctx.read_resource()
# ---------------------------------------------------------------------------
_TOOL_SKILL_MAP = {
    # wifi-pentest
    "airmon_ng": "wifi-pentest", "airodump_ng": "wifi-pentest",
    "aireplay_ng": "wifi-pentest", "aircrack_ng": "wifi-pentest",
    "hcxdumptool": "wifi-pentest", "wifite": "wifi-pentest",
    "wifite2": "wifi-pentest",
    # nmap-recon
    "nmap": "nmap-recon", "nmap_advanced": "nmap-recon",
    "masscan": "nmap-recon", "rustscan": "nmap-recon", "arp_scan": "nmap-recon",
    "autorecon": "nmap-recon",
    # subdomain-enum
    "subfinder": "subdomain-enum", "amass": "subdomain-enum",
    "dnsenum": "subdomain-enum", "fierce": "subdomain-enum",
    "theharvester": "subdomain-enum",
    # osint-recon
    "whois": "osint-recon", "sherlock": "osint-recon",
    "spiderfoot": "osint-recon", "sublist3r": "osint-recon",
    "parsero": "osint-recon",
    # web-recon
    "wafw00f": "web-recon", "httpx": "web-recon", "katana": "web-recon",
    "gobuster": "web-recon", "ffuf": "web-recon", "feroxbuster": "web-recon",
    "dirsearch": "web-recon", "wpscan": "web-recon", "testssl": "web-recon",
    "whatweb": "web-recon", "joomscan": "web-recon", "hakrawler": "web-recon",
    "gau": "web-recon", "waybackurls": "web-recon", "arjun": "web-recon",
    "paramspider": "web-recon", "x8": "web-recon", "anew": "web-recon",
    "uro": "web-recon",
    # web-vuln
    "nuclei": "web-vuln", "nikto": "web-vuln", "sqlmap": "web-vuln",
    "dalfox": "web-vuln", "xsser": "web-vuln", "dotdotpwn": "web-vuln",
    "jaeles": "web-vuln", "commix": "web-vuln", "vulnx": "web-vuln",
    "zap": "web-vuln",
    # password-cracking
    "hashid": "password-cracking", "john": "password-cracking",
    "hashcat": "password-cracking", "hydra": "password-cracking",
    "medusa": "password-cracking", "ophcrack": "password-cracking",
    "patator": "password-cracking",
    # smb-enum
    "nbtscan": "smb-enum", "smbmap": "smb-enum", "enum4linux": "smb-enum",
    "netexec": "smb-enum", "rpcclient": "smb-enum",
    # exploitation
    "metasploit": "exploitation", "msfvenom": "exploitation",
    "exploit_db": "exploitation", "searchsploit": "exploitation",
    # binary-analysis
    "checksec": "binary-analysis", "strings": "binary-analysis",
    "binwalk": "binary-analysis", "radare2": "binary-analysis",
    "ropgadget": "binary-analysis", "ropper": "binary-analysis",
    "one_gadget": "binary-analysis", "gdb": "binary-analysis",
    # cloud-audit
    "prowler": "cloud-audit", "trivy": "cloud-audit",
    "kube_hunter": "cloud-audit", "kube_bench": "cloud-audit",
    "checkov": "cloud-audit", "terrascan": "cloud-audit",
    # active-directory
    "impacket": "active-directory", "ldapdomaindump": "active-directory",
    "adidnsdump": "active-directory", "certipy": "active-directory",
    "certipy_ad": "active-directory", "mitm6": "active-directory",
    "pywerview": "active-directory", "bloodhound": "active-directory",
    "bloodhound_python": "active-directory",
}

_SKILL_SUPPORT_FILES = ("REFERENCE.md", "CHECKLIST.md", "EXAMPLES.md")
_TOOL_REGISTRY_ALIASES = {
    "aircrack_ng": "aircrack-ng",
    "airmon_ng": "airmon-ng",
    "airodump_ng": "airodump-ng",
    "aireplay_ng": "aireplay-ng",
    "arp_scan": "arp-scan",
    "kube_hunter": "kube-hunter",
    "one_gadget": "one-gadget",
    "theharvester": "theHarvester",
}


async def _read_skill_document(ctx: Context, skill_name: str, filename: str) -> Optional[str]:
    """Best-effort resource read for one file inside a skill bundle."""
    try:
        resource = await ctx.read_resource(f"skill://{skill_name}/{filename}")
    except Exception:
        return None

    contents = getattr(resource, "contents", None)
    if not contents:
        return None

    first = contents[0] if isinstance(contents, list) else contents
    raw = getattr(first, "content", None)
    if raw is None:
        return None

    return raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")


async def _read_skill_bundle(ctx: Context, skill_name: str) -> Dict[str, str]:
    """Load the main skill file plus common supporting files, if present."""
    documents: Dict[str, str] = {}
    for filename in ("SKILL.md",) + _SKILL_SUPPORT_FILES:
        text = await _read_skill_document(ctx, skill_name, filename)
        if text:
            documents[filename] = text
    return documents


def _get_registry_tool_definition(tool_name: str) -> Optional[Dict[str, Any]]:
    """Resolve a direct tool name to a compact registry schema if one exists."""
    tool_def = get_tool(tool_name)
    if tool_def:
        return tool_def

    alias = _TOOL_REGISTRY_ALIASES.get(tool_name)
    if alias:
        return get_tool(alias)
    return None


def _infer_param_type(default: Any) -> type:
    """Infer a simple Python type from a default value."""
    if isinstance(default, bool):
        return bool
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    if isinstance(default, dict):
        return dict
    if isinstance(default, list):
        return list
    return str


def _resolve_required_param_type(spec: Dict[str, Any]) -> type:
    """Resolve a required parameter type from optional registry metadata."""
    declared_type = spec.get("type")
    if declared_type == "bool":
        return bool
    if declared_type == "int":
        return int
    if declared_type == "float":
        return float
    if declared_type == "dict":
        return dict
    if declared_type == "list":
        return list
    return str


def _build_typed_tool_doc(tool_name: str, description: str, tool_def: Dict[str, Any]) -> str:
    """Generate a concise docstring for a typed wrapper tool."""
    lines = [description, "", "Args:"]
    for param_name, spec in tool_def.get("params", {}).items():
        annotation = _resolve_required_param_type(spec)
        lines.append(f"    {param_name}: Required parameter")
    for param_name, default in tool_def.get("optional", {}).items():
        lines.append(f"    {param_name}: Optional parameter. Default: {default!r}")
    lines.extend(
        [
            "",
            "Returns:",
            "    Tool execution results",
            "",
            "Notes:",
            f"    This is a typed wrapper over run_security_tool(tool_name={tool_name!r}, parameters=...).",
        ]
    )
    return "\n".join(lines)


def _create_typed_tool_wrapper(tool_name: str, tool_def: Dict[str, Any], run_security_tool):
    """Create a typed MCP tool wrapper around run_security_tool()."""
    required_params = list(tool_def.get("params", {}).keys())
    optional_params = tool_def.get("optional", {})
    annotations: Dict[str, Any] = {"return": Dict[str, Any]}

    async def typed_tool(**kwargs) -> Dict[str, Any]:
        ctx = get_context()
        payload = {name: kwargs[name] for name in required_params}
        for name in optional_params:
            if name in kwargs:
                payload[name] = kwargs[name]
        return await run_security_tool(ctx, tool_name, payload)

    parameters = []
    for name in required_params:
        annotation = _resolve_required_param_type(tool_def["params"][name])
        annotations[name] = annotation
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
            )
        )
    for name, default in optional_params.items():
        annotation = _infer_param_type(default)
        annotations[name] = annotation
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )

    typed_tool.__name__ = f"{tool_name.replace('-', '_')}_typed"
    typed_tool.__doc__ = _build_typed_tool_doc(tool_name, tool_def["desc"], tool_def)
    typed_tool.__annotations__ = annotations
    typed_tool.__signature__ = inspect.Signature(parameters=parameters)
    return typed_tool


def _register_skills(mcp: FastMCP, logger) -> None:
    """Mount the local skills/ directory as MCP resources if it exists."""
    if SkillsDirectoryProvider is None:
        logger.warning("fastmcp SkillsDirectoryProvider not available; skipping skills registration")
        return
    skills_dir = Path(__file__).parent.parent / "skills"
    if not skills_dir.exists():
        return
    mcp.add_provider(
        SkillsDirectoryProvider(
            roots=skills_dir,
            supporting_files="resources",
            reload=True,
        )
    )
    logger.info("🤖 Skills initialized (supporting files exposed as resources, reload enabled)")

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
    mcp = FastMCP("hexstrike-ai-pulse", transforms=transforms)

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
    mcp = FastMCP("hexstrike-ai pulse", transforms=transforms)

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
    from mcp_core.testssl_direct import testssl_exec
    from mcp_core.web_probe_direct import web_probe_exec
    from mcp_core.vuln_intel_direct import vuln_intel_exec

    DIRECT_TOOLS = {
        # wifi
        "airmon_ng":         (wifi_exec, "airmon_ng"),
        "airodump_ng":       (wifi_exec, "airodump_ng"),
        "aireplay_ng":       (wifi_exec, "aireplay_ng"),
        "aircrack_ng":       (wifi_exec, "aircrack_ng"),
        "hcxdumptool":       (wifi_exec, "hcxdumptool"),
        "wifite":            (wifi_exec, "wifite2"),
        "wifite2":           (wifi_exec, "wifite2"),
        # recon
        "amass":             (recon_exec, "amass"),
        "subfinder":         (recon_exec, "subfinder"),
        "autorecon":         (recon_exec, "autorecon"),
        "theharvester":      (recon_exec, "theharvester"),
        "dnsenum":           (recon_exec, "dnsenum"),
        "fierce":            (recon_exec, "fierce"),
        "whois":             (recon_exec, "whois"),
        # net_scan
        "nmap":              (net_scan_exec, "nmap"),
        "nmap_advanced":     (net_scan_exec, "nmap-advanced"),
        "masscan":           (net_scan_exec, "masscan"),
        "rustscan":          (net_scan_exec, "rustscan"),
        "arp_scan":          (net_scan_exec, "arp-scan"),
        # web_scan
        "nikto":             (web_scan_exec, "nikto"),
        "sqlmap":            (web_scan_exec, "sqlmap"),
        "wpscan":            (web_scan_exec, "wpscan"),
        "dalfox":            (web_scan_exec, "dalfox"),
        "jaeles":            (web_scan_exec, "jaeles"),
        "xsser":             (web_scan_exec, "xsser"),
        "zap":               (web_scan_exec, "zap"),
        # web_fuzz
        "gobuster":          (web_fuzz_exec, "gobuster"),
        "ffuf":              (web_fuzz_exec, "ffuf"),
        "feroxbuster":       (web_fuzz_exec, "feroxbuster"),
        "dirsearch":         (web_fuzz_exec, "dirsearch"),
        "dirb":              (web_fuzz_exec, "dirb"),
        "wfuzz":             (web_fuzz_exec, "wfuzz"),
        "dotdotpwn":         (web_fuzz_exec, "dotdotpwn"),
        # password_cracking
        "hydra":             (pwdcrack_exec, "hydra"),
        "hashcat":           (pwdcrack_exec, "hashcat"),
        "john":              (pwdcrack_exec, "john"),
        "medusa":            (pwdcrack_exec, "medusa"),
        "patator":           (pwdcrack_exec, "patator"),
        "hashid":            (pwdcrack_exec, "hashid"),
        "ophcrack":          (pwdcrack_exec, "ophcrack"),
        # smb_enum
        "enum4linux":        (smb_enum_exec, "enum4linux"),
        "netexec":           (smb_enum_exec, "netexec"),
        "rpcclient":         (smb_enum_exec, "rpcclient"),
        "smbmap":            (smb_enum_exec, "smbmap"),
        "nbtscan":           (smb_enum_exec, "nbtscan"),
        # exploit
        "metasploit":        (exploit_exec, "metasploit"),
        "msfvenom":          (exploit_exec, "msfvenom"),
        "searchsploit":      (exploit_exec, "exploit_db"),
        "exploit_db":        (exploit_exec, "exploit_db"),
        # web_recon
        "katana":            (web_recon_exec, "katana"),
        "hakrawler":         (web_recon_exec, "hakrawler"),
        "gau":               (web_recon_exec, "gau"),
        "waybackurls":       (web_recon_exec, "waybackurls"),
        "httpx":             (web_recon_exec, "httpx"),
        "wafw00f":           (web_recon_exec, "wafw00f"),
        "arjun":             (web_recon_exec, "arjun"),
        "paramspider":       (web_recon_exec, "paramspider"),
        "x8":                (web_recon_exec, "x8"),
        # security
        "prowler":           (security_exec, "prowler"),
        "trivy":             (security_exec, "trivy"),
        "kube_hunter":       (security_exec, "kube-hunter"),
        "kube_bench":        (security_exec, "kube-bench"),
        "checkov":           (security_exec, "checkov"),
        "terrascan":         (security_exec, "terrascan"),
        # misc
        "ropgadget":         (misc_exec, "ropgadget"),
        "ropper":            (misc_exec, "ropper"),
        "one_gadget":        (misc_exec, "one_gadget"),
        "volatility":        (misc_exec, "volatility"),
        "volatility3":       (misc_exec, "volatility3"),
        "gdb":               (misc_exec, "gdb"),
        "radare2":           (misc_exec, "radare2"),
        "strings":           (misc_exec, "strings"),
        "objdump":           (misc_exec, "objdump"),
        "checksec":          (misc_exec, "checksec"),
        "binwalk":           (misc_exec, "binwalk"),
        "ghidra":            (misc_exec, "ghidra"),
        "angr":              (misc_exec, "angr"),
        "xxd":               (misc_exec, "xxd"),
        "mysql":             (misc_exec, "mysql"),
        "sqlite":            (misc_exec, "sqlite"),
        "exiftool":          (misc_exec, "exiftool"),
        "foremost":          (misc_exec, "foremost"),
        "steghide":          (misc_exec, "steghide"),
        "hashpump":          (misc_exec, "hashpump"),
        "anew":              (misc_exec, "anew"),
        "uro":               (misc_exec, "uro"),
        "nuclei":            (misc_exec, "nuclei"),
        "responder":         (misc_exec, "responder"),
        # osint
        "sherlock":          (osint_exec, "sherlock"),
        "spiderfoot":        (osint_exec, "spiderfoot"),
        "sublist3r":         (osint_exec, "sublist3r"),
        "parsero":           (osint_exec, "parsero"),
        # web_probe
        "testssl":           (testssl_exec, "testssl"),
        "whatweb":           (web_probe_exec, "whatweb"),
        "commix":            (web_probe_exec, "commix"),
        "joomscan":          (web_probe_exec, "joomscan"),
        # vuln_intel
        "vulnx":             (vuln_intel_exec, "vulnx"),
        # active_directory
        "impacket":          (ad_exec, "impacket"),
        "ldapdomaindump":    (ad_exec, "ldapdomaindump"),
        "adidnsdump":        (ad_exec, "adidnsdump"),
        "certipy":           (ad_exec, "certipy_ad"),
        "certipy_ad":        (ad_exec, "certipy_ad"),
        "mitm6":             (ad_exec, "mitm6"),
        "pywerview":         (ad_exec, "pywerview"),
        "bloodhound":        (ad_exec, "bloodhound"),
        "bloodhound_python": (ad_exec, "bloodhound"),
    }

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
        _t_start = time.time()
        _telemetry: Dict[str, Any] = {
            "tool":            tool_name,
            "success":         False,
            "duration":        0.0,
            "timeout":         False,
            "cache_hit":       False,
            "session_state":   False,
            "confirmation":    None,   # None | "accepted" | "denied" | "skipped"
            "opt_profile":     "normal",
            "skill_injected":  False,
            "prompt_suggested": False,
        }
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

        # 1. Resolve target once — needed by elicitation, tech detect, get_prompt, and cache
        target = (
            params.get("target")
            or params.get("url")
            or params.get("domain")
            or params.get("interface", "")
        )

        # 2. Elicitation — destructive tools require explicit user confirmation
        destructive_request = _build_destructive_confirmation(tool_name, params)
        if destructive_request:
            confirmed = await confirm_destructive_action(
                ctx,
                action=destructive_request["action"],
                detail=destructive_request["detail"],
                warning=destructive_request["warning"],
            )
            if not confirmed:
                _telemetry["confirmation"] = "denied"
                _telemetry["duration"] = round(time.time() - _t_start, 3)
                logger.info("[telemetry] %s", json.dumps(_telemetry))
                return {
                    "success": False,
                    "error": f"Action cancelled - {destructive_request['action']} requires explicit confirmation",
                }
            _telemetry["confirmation"] = "accepted"
        else:
            _telemetry["confirmation"] = "skipped" if not destructive_request else None
        # 3. Skill guidance
        skill_name = _TOOL_SKILL_MAP.get(tool_name.lower())
        if skill_name:
            skill_doc = await _read_skill_document(ctx, skill_name, "SKILL.md")
            if skill_doc:
                lines = [l for l in skill_doc.splitlines() if l.strip() and not l.startswith("---")]
                header = "\n".join(lines[:4])
                await ctx.info(f"📚 [{skill_name}] {header}")
                _telemetry["skill_injected"] = True

        # ParameterOptimizer — enrich params before execution
        # Caller can pass _profile (stealth/normal/aggressive) and _tech (dict) in params
        opt_profile = params.pop("_profile", "normal")
        tech_dict   = params.pop("_tech", None)
        tech_profile: Optional[TechProfile] = None

        if isinstance(tech_dict, dict):
            # Caller passed explicit tech info
            try:
                tech_profile = TechProfile(
                    web_servers = tech_dict.get("web_servers", []),
                    frameworks  = tech_dict.get("frameworks", []),
                    cms         = tech_dict.get("cms", []),
                    databases   = tech_dict.get("databases", []),
                    languages   = tech_dict.get("languages", []),
                    security    = tech_dict.get("security", []),
                    services    = tech_dict.get("services", []),
                )
            except Exception:
                pass
        else:
            # Auto-detect: session state first, then scan cache
            if target:
                # 1. Try session-persisted TechProfile (fastest — no recompute)
                try:
                    cached_dict = await ctx.get_state(f"tech:{target}")
                    if cached_dict and isinstance(cached_dict, dict):
                        tech_profile = TechProfile(
                            web_servers = cached_dict.get("web_servers", []),
                            frameworks  = cached_dict.get("frameworks", []),
                            cms         = cached_dict.get("cms", []),
                            databases   = cached_dict.get("databases", []),
                            languages   = cached_dict.get("languages", []),
                            security    = cached_dict.get("security", []),
                            services    = cached_dict.get("services", []),
                        )
                        _telemetry["session_state"] = True
                        await ctx.info(f"🧠 Tech profile restored from session: {tech_profile.summary()}")
                except Exception:
                    pass

                # 2. Fall back to scan cache if no session state
                if tech_profile is None:
                    tech_profile = _detect_from_cache(target)
                    if tech_profile:
                        await ctx.info(f"🧠 Tech detected from cache: {tech_profile.summary()}")
                        # Persist to session state for subsequent calls
                        try:
                            await ctx.set_state(f"tech:{target}", {
                                "web_servers": tech_profile.web_servers,
                                "frameworks":  tech_profile.frameworks,
                                "cms":         tech_profile.cms,
                                "databases":   tech_profile.databases,
                                "languages":   tech_profile.languages,
                                "security":    tech_profile.security,
                                "services":    tech_profile.services,
                            })
                        except Exception:
                            pass  # session state unavailable — no-op

        # ctx.get_prompt() — suggest workflow prompt based on TechProfile
        if tech_profile:
            suggested_prompt = None
            if tech_profile.cms and any(c in tech_profile.cms for c in ("wordpress", "joomla", "drupal")):
                suggested_prompt = ("bug_bounty_recon", {"target": target or tool_name})
            elif tech_profile.security and any(s in tech_profile.security for s in ("cloudflare", "akamai", "aws_waf")):
                suggested_prompt = ("bug_bounty_recon", {"target": target or tool_name})
            elif "smb" in (tech_profile.services or []) or "ldap" in (tech_profile.services or []):
                suggested_prompt = ("smb_lateral_movement", {"target": target or "unknown"})
            elif any(s in (tech_profile.services or []) for s in ("ssh", "rdp")) and not target:
                pass  # not enough context

            if suggested_prompt:
                try:
                    prompt_name, prompt_args = suggested_prompt
                    prompt_result = await ctx.get_prompt(prompt_name, prompt_args)
                    if prompt_result and prompt_result.messages:
                        first_msg = prompt_result.messages[0]
                        hint = getattr(getattr(first_msg, 'content', None), 'text', '')
                        if hint:
                            await ctx.info(f"💡 Workflow suggestion [{prompt_name}]: {hint[:120]}")
                            _telemetry["prompt_suggested"] = True
                except Exception:
                    pass  # prompt unavailable — no-op

        # Rate limit profile from session state — override normal profile
        if not opt_profile or opt_profile == "normal":
            try:
                saved_rl = await ctx.get_state(f"ratelimit:{target}")
                if saved_rl:
                    opt_profile = saved_rl
                    _telemetry["opt_profile"] = opt_profile
                    await ctx.info(f"⚡ Rate limit profile restored: {opt_profile}")
            except Exception:
                pass

        _telemetry["opt_profile"] = opt_profile
        params = _optimizer.optimize(tool_name.lower(), params, tech_profile, opt_profile)
        optimizer_meta = params.pop("_optimizer", {})
        if optimizer_meta.get("forced_stealth"):
            await ctx.info(f"🛡️ WAF detected → stealth mode forced for {tool_name}")
        elif opt_profile != "normal":
            await ctx.info(f"⚙️ Profile: {optimizer_meta.get('profile', opt_profile)}")

        loop = asyncio.get_running_loop()
        future = asyncio.ensure_future(
            loop.run_in_executor(None, lambda: exec_func(tool_key, params))
        )

        await ctx.report_progress(0, 100)
        phases = [(25, "🔧 Preparing..."), (50, "⚙️  Running..."), (75, "📊 Processing results...")]
        tick = 12
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=tick)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info(f"✅ {tool_name} completed")
            _telemetry["success"] = True

            # Rate limit detection on successful results
            output = str(result.get("output", "") or result.get("data", ""))
            status_code = result.get("returncode", 0) or 0
            rl = _rate_limiter.detect_rate_limiting(output, status_code)
            if rl["detected"]:
                recommended = rl["recommended_profile"]
                _telemetry["rate_limit"] = recommended
                await ctx.info(f"⚠️ Rate limit detected (confidence={rl['confidence']:.0%}) → switching to {recommended}")
                try:
                    await ctx.set_state(f"ratelimit:{target}", recommended)
                except Exception:
                    pass

            if target:
                cache_key = f"{tool_name}:{target}"
                exec_time = result.get("execution_time", 0.0)
                _scan_cache.set(cache_key, {
                    "tool": tool_name, "target": target,
                    "result": result, "timestamp": time.time(),
                }, execution_time=exec_time)
        else:
            await ctx.error(f"❌ {tool_name} failed: {result.get('error', 'unknown')}")

        _telemetry["duration"] = round(time.time() - _t_start, 3)
        logger.info("[telemetry] %s", json.dumps(_telemetry))
        return result

    @mcp.tool(
        description="Return the local skill bundle associated with a HexStrike tool",
        annotations={"readOnlyHint": True, "openWorldHint": False},
    )
    async def get_tool_skill(
        ctx: Context,
        tool_name: str,
    ) -> Dict[str, Any]:
        """Fetch the local skill documents mapped to a given tool."""
        skill_name = _TOOL_SKILL_MAP.get(tool_name.lower())
        if not skill_name:
            return {
                "success": False,
                "error": f"No skill mapping found for tool: {tool_name}",
            }

        documents = await _read_skill_bundle(ctx, skill_name)
        if not documents:
            return {
                "success": False,
                "error": f"Skill bundle not readable for: {skill_name}",
                "skill_name": skill_name,
            }

        return {
            "success": True,
            "tool_name": tool_name,
            "skill_name": skill_name,
            "documents": documents,
            "available_files": sorted(documents),
        }

    # ========================================================================
    # plan_attack — IntelligentDecisionEngine MCP tool
    # ========================================================================
    from server_core.intelligence.intelligent_decision_engine import IntelligentDecisionEngine as _IDE
    _ide = _IDE()

    @mcp.tool(
        description="Analyze a target and generate an intelligent attack chain with ordered steps, tool selection and success probabilities",
        annotations={"readOnlyHint": False, "openWorldHint": True},
    )
    async def plan_attack(
        ctx: Context,
        target: str,
        objective: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        Generate an intelligent attack chain for a target.

        Args:
            target:    Target (IP, URL, domain, file path)
            objective: comprehensive | quick | stealth | ctf | bug_bounty_recon |
                       bug_bounty_hunting | aws | kubernetes | containers | iac

        Returns:
            AttackChain dict with ordered steps, tools, probabilities, estimated time
        """
        await ctx.info(f"🧠 Analyzing target: {target} (objective={objective})")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()

        # Check session state first — avoid re-analyzing if profile already exists
        profile = None
        try:
            saved_profile = await ctx.get_state(f"ide_profile:{target}")
            if saved_profile and isinstance(saved_profile, dict):
                from shared.target_profile import TargetProfile as _TargetProfile
                profile = _TargetProfile.from_dict(saved_profile)
                await ctx.info(f"⚡ Target profile restored from session — skipping re-analysis")
                await ctx.report_progress(50, 100)
        except Exception:
            pass

        if profile is None:
            profile = await loop.run_in_executor(None, lambda: _ide.analyze_target(target))
            await ctx.report_progress(50, 100)
            await ctx.info(f"🎯 Target type: {profile.target_type.value} | Risk: {profile.risk_level} | Confidence: {profile.confidence_score:.0%}")
            # Persist to session state
            try:
                await ctx.set_state(f"ide_profile:{target}", profile.to_dict())
            except Exception:
                pass
        else:
            await ctx.info(f"🎯 Target type: {profile.target_type.value} | Risk: {profile.risk_level} | Confidence: {profile.confidence_score:.0%}")

        chain = await loop.run_in_executor(None, lambda: _ide.create_attack_chain(profile, objective))
        await ctx.report_progress(100, 100)
        await ctx.info(f"✅ Attack chain ready: {len(chain.steps)} steps | Risk: {chain.risk_level}")
        return chain.to_dict()

    typed_tools_registered = 0
    for public_name in sorted(DIRECT_TOOLS):
        tool_def = _get_registry_tool_definition(public_name)
        if not tool_def:
            continue
        wrapper = _create_typed_tool_wrapper(public_name, tool_def, run_security_tool)
        mcp.tool(
            name=public_name,
            description=tool_def["desc"],
            annotations={"readOnlyHint": False, "openWorldHint": True},
        )(wrapper)
        typed_tools_registered += 1

    logger.info(
        f"🚀 Phase 3: {len(DIRECT_TOOLS)} direct routes, {typed_tools_registered} typed MCP tools, + skill bundle helper — no Flask"
    )

    # ========================================================================
    # Resources MCP — health + scan results
    # ========================================================================

    @mcp.resource("health://server")
    async def server_health() -> str:
        """Server health and runtime statistics."""
        uptime = int(time.time() - _server_start_time)
        return json.dumps({
            "status":         "healthy",
            "server":         "hexstrike-ai-pulse",
            "fastmcp":        "3.2.4",
            "uptime_seconds": uptime,
            "tools_count":    len(DIRECT_TOOLS),
            "cached_scans":   len(_scan_cache),
            "cache_stats":    _scan_cache.stats(),
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
                "target":  target,
                "status":  "no_results",
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
                "target":  target,
                "tool":    tool_name,
                "status":  "no_results",
                "message": f"No cached result for {tool_name} on {target}",
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
