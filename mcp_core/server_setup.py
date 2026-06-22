import asyncio
import hashlib
import inspect
import json
import logging
import os
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_context
from fastmcp.server.tasks import TaskConfig
from mcp_core.parameter_optimizer import ParameterOptimizer
from mcp_core.technology_detector import TechProfile, TechnologyDetector
from mcp_core.elicitation import confirm_destructive_action
from server_core.rate_limit_detector import RateLimitDetector
from server_core.operational_metrics import _op_metrics
from server_core.singletons import get_tool_stats_store, get_target_store
from server_core.hexstrike_middleware import HexStrikeLoggingMiddleware, HexStrikeSessionMiddleware
from server_core.request_context import get_request_id
from tool_registry import get_tool
from mcp_core.tool_routes import TOOL_ROUTES
from mcp_core.tool_registry_v2 import _registry
from server_core.telemetry_pipeline import _pipeline

import importlib

# Module-level cache for DIRECT_TOOLS — stores (mod_path, func_name, binary) tuples
_DIRECT_TOOLS_CACHE: dict = {}
_MOCK_EXECUTORS: dict = {}

def _resolve_exec_func(module_path: str, func_name: str):
    """Dynamically resolve an executor function — enables mock patching at call time."""
    mock = _MOCK_EXECUTORS.get(func_name)
    if mock is not None:
        return mock
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name, None)
    except ImportError:
        return None

def _get_direct_tools() -> dict:
    """Return DIRECT_TOOLS (tool exec functions), lazy-populated on first call."""
    if not _DIRECT_TOOLS_CACHE:
        _populate_direct_tools()
    return _DIRECT_TOOLS_CACHE

def _populate_direct_tools() -> None:
    """Populate _DIRECT_TOOLS_CACHE from TOOL_ROUTES — safe to call any time, idempotent."""
    global _DIRECT_TOOLS_CACHE
    if _DIRECT_TOOLS_CACHE:
        return
    for tool_name, (mod_path, func_name, binary) in TOOL_ROUTES.items():
        if _resolve_exec_func(mod_path, func_name) is not None:
            _DIRECT_TOOLS_CACHE[tool_name] = (mod_path, func_name, binary)

try:
    from fastmcp.server.providers.skills import SkillsDirectoryProvider
except ImportError:
    SkillsDirectoryProvider = None

try:
    from fastmcp.server.transforms.search import RegexSearchTransform, serialize_tools_for_output_markdown
except ImportError:
    RegexSearchTransform = None
    serialize_tools_for_output_markdown = None

# ---------------------------------------------------------------------------
# In-memory scan result cache — populated by run_security_tool
# ---------------------------------------------------------------------------
from server_core.advanced_cache import AdvancedCache as _AdvancedCache


class _ScanCache(_AdvancedCache):
    """Scan-specific cache: wraps AdvancedCache with adaptive TTL from execution_time + learning."""
    _TTL_DEFAULT = 1800   # 30 min
    _TTL_MEDIUM  = 3600   # 60 min (exec > 10s)
    _TTL_LONG    = 5400   # 90 min (exec > 60s)
    _TTL_MIN     = 300    # 5 min floor
    _TTL_MAX     = 7200   # 2h ceiling

    def __init__(self, max_size: int = 500, default_ttl: int = 1800) -> None:
        super().__init__(max_size, default_ttl)
        self._ttl_scores: dict[str, dict[str, float | int]] = {}
        self._ttl_lock = threading.RLock()

    def set(self, key: str, value: Any, execution_time: float = 0.0, ttl: Optional[int] = None) -> None:  # type: ignore[override]
        tool = value.get("tool", "unknown") if isinstance(value, dict) else "unknown"
        if ttl is None:
            ttl = self._get_adaptive_ttl(tool, execution_time)
        with self._ttl_lock:
            entry = self._ttl_scores.setdefault(tool, {
                "sets": 0, "hits": 0, "misses": 0, "current_ttl": float(ttl),
            })
            entry["sets"] = int(entry["sets"]) + 1  # type: ignore[assignment]
            entry["current_ttl"] = float(ttl)
        super().set(key, value, ttl=ttl)

    def get(self, key: str) -> Any:
        result = super().get(key)
        tool = "unknown"
        if isinstance(key, str):
            parts = key.split(":")
            tool = parts[1] if len(parts) >= 2 else "unknown"
        with self._ttl_lock:
            entry = self._ttl_scores.setdefault(tool, {
                "sets": 0, "hits": 0, "misses": 0, "current_ttl": float(self._TTL_DEFAULT),
            })
            if result is not None:
                entry["hits"] = int(entry["hits"]) + 1  # type: ignore[assignment]
            else:
                entry["misses"] = int(entry["misses"]) + 1  # type: ignore[assignment]
        return result

    def _get_adaptive_ttl(self, tool: str, execution_time: float) -> int:
        with self._ttl_lock:
            entry = self._ttl_scores.get(tool)
            if entry and int(entry["sets"]) > 2:  # need 3+ samples before adapting
                total = int(entry["hits"]) + int(entry["misses"])
                hit_ratio = int(entry["hits"]) / (total + 0.001)
                current = float(entry["current_ttl"])
                if hit_ratio > 0.3 and current < self._TTL_MAX:
                    return int(min(current * 1.2, self._TTL_MAX))
                if hit_ratio < 0.05 and current > self._TTL_MIN:
                    return int(max(current * 0.8, self._TTL_MIN))
                return int(current)
        if execution_time > 60:
            return self._TTL_LONG
        if execution_time > 10:
            return self._TTL_MEDIUM
        return self._TTL_DEFAULT

    def get_ttl_scores(self) -> dict[str, dict[str, Any]]:
        with self._ttl_lock:
            result: dict[str, dict[str, Any]] = {}
            for tool, e in sorted(self._ttl_scores.items()):
                total = int(e["hits"]) + int(e["misses"])
                hit_ratio = round(int(e["hits"]) / (total + 0.001), 3)
                result[tool] = {
                    "sets": int(e["sets"]),
                    "hits": int(e["hits"]),
                    "misses": int(e["misses"]),
                    "hit_ratio": hit_ratio,
                    "current_ttl_seconds": int(e["current_ttl"]),
                }
            return result

    def stats(self) -> Dict[str, Any]:
        base = self.get_stats()
        base["ttl_scores"] = self.get_ttl_scores()
        return base


_scan_cache = _ScanCache(max_size=500, default_ttl=1800)
_server_start_time = time.time()
_optimizer    = ParameterOptimizer()

# Per-tool subprocess timeouts (seconds) — used by scan_background() and scan()
# Each value should be realistic for the tool: nmap ~60s, nikto/sqlmap ~480s.
# opencode MCP timeout (600s) > per-tool max > EnhancedCommandExecutor internal timeout (COMMAND_TIMEOUT=300s)
TOOL_TIMEOUTS: dict[str, int] = {
    "nmap": 60,
    "whatweb": 30,
    "nikto": 480,
    "sqlmap": 480,
    "gobuster": 300,
    "nuclei": 180,
}
_detector     = TechnologyDetector()
_rate_limiter = RateLimitDetector()
_rate_limit_events: list[dict] = []  # appended by run_security_tool on detection
_PRIMITIVE_TOOLS = {"execute_code", "http_request", "raw_tcp"}

logger = logging.getLogger(__name__)


def _cache_key_for(session_id: str, tool_name: str, target: str, params: Dict[str, Any]) -> str:
    relevant = {
        k: v for k, v in sorted(params.items())
        if not k.startswith("_") and k not in ("target",)
    }
    if not relevant:
        return f"{session_id}:{tool_name}:{target}"
    param_str = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
    return f"{session_id}:{tool_name}:{target}:{param_hash}"


def _collect_cached_scans(session_id: str, target: str) -> Dict[str, Any]:
    """Collect all cached scan results for a given target in the current session."""
    scans: Dict[str, Any] = {}
    for k, v in _scan_cache.items():
        if (k.startswith(f"{session_id}:") or k.startswith("seed:")) and v.get("target") == target:
            tool = v.get("tool")
            if tool:
                scans[tool] = v.get("result", {})
    return scans


def _enrich_profile_from_cache(profile: Any, cached_scans: Dict[str, Any]) -> Any:
    """Inject cached scan results into a TargetProfile for better attack planning."""
    from shared.target_types import TechnologyStack

    # nmap — ports and services
    nmap_result = cached_scans.get("nmap") or cached_scans.get("nmap_advanced")
    if nmap_result:
        output = str(nmap_result.get("output", "") or nmap_result.get("stdout", ""))
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and "/" in parts[0]:
                try:
                    port = int(parts[0].split("/")[0])
                    if port not in profile.open_ports:
                        profile.open_ports.append(port)
                    if len(parts) >= 3 and port not in profile.services:
                        profile.services[port] = parts[2]
                except ValueError:
                    pass
        if profile.open_ports:
            profile.confidence_score = min(1.0, profile.confidence_score + 0.2)

    # whatweb — technology detection
    whatweb_result = cached_scans.get("whatweb")
    if whatweb_result:
        output = str(whatweb_result.get("output", "") or whatweb_result.get("stdout", ""))
        tech_keywords = {
            "wordpress": TechnologyStack.WORDPRESS,
            "joomla":    TechnologyStack.JOOMLA,
            "drupal":    TechnologyStack.DRUPAL,
            "nginx":     TechnologyStack.NGINX,
            "apache":    TechnologyStack.APACHE,
            "php":       TechnologyStack.PHP,
            "python":    TechnologyStack.PYTHON,
            "node.js":   TechnologyStack.NODEJS,
            "java":      TechnologyStack.JAVA,
            "react":     TechnologyStack.REACT,
            "angular":   TechnologyStack.ANGULAR,
            "vue":       TechnologyStack.VUE,
        }
        output_lower = output.lower()
        for keyword, tech in tech_keywords.items():
            if keyword in output_lower and tech not in profile.technologies:
                profile.technologies.append(tech)
        profile.confidence_score = min(1.0, profile.confidence_score + 0.1)

    # wafw00f — WAF detection boosts confidence
    if cached_scans.get("wafw00f"):
        profile.confidence_score = min(1.0, profile.confidence_score + 0.05)

    # testssl — SSL/TLS info
    testssl_result = cached_scans.get("testssl")
    if testssl_result:
        output = str(testssl_result.get("output", "") or testssl_result.get("stdout", ""))
        if output and not profile.ssl_info:
            profile.ssl_info = {"source": "testssl", "summary": output[:200]}
        profile.confidence_score = min(1.0, profile.confidence_score + 0.05)

    # Adjust attack surface and risk based on real port data
    if profile.open_ports:
        profile.attack_surface_score = min(10.0, profile.attack_surface_score + len(profile.open_ports) * 0.5)
        if len(profile.open_ports) > 5:
            profile.risk_level = "high"
        elif len(profile.open_ports) > 2:
            profile.risk_level = "medium"

    return profile


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
        entry = next(
            (v for k, v in _scan_cache.items()
             if v.get("tool") == tool and v.get("target") == target),
            None,
        )
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
# Per-category timeout references (for documentation only)
# Actual execution timeout is handled by EnhancedCommandExecutor internally.
# FastMCP mcp.tool(timeout=None) disables the framework-level timeout so
# that CancelledError never fires — the subprocess owns its own lifecycle.
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

_SKILL_SUPPORT_FILES = ("REFERENCE.md",)
_TOOL_REGISTRY_ALIASES = {
    "aircrack_ng": "aircrack-ng",
    "airmon_ng": "airmon-ng",
    "airodump_ng": "airodump-ng",
    "aireplay_ng": "aireplay-ng",
    "arp_scan": "arp-scan",
    "bettercap_wifi": "bettercap",
    "airbase_ng": "airbase-ng",
    "airdecap_ng": "airdecap-ng",
    "api_schema_analyzer": "api-schema-analyzer",
    "evil_winrm": "evil-winrm",
    "graphql_scanner": "graphql-scanner",
    "jwt_analyzer": "jwt-analyzer",
    "kube_hunter": "kube-hunter",
    "libc": "libc-database",
    "one_gadget": "one-gadget",
    "theharvester": "theHarvester",
    "wifite2": "wifite",
    "http_request": "http-framework",
    "tcp_send": "tcp-send",
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


def _normalize_tool_result(result: Any) -> Dict[str, Any]:
    """Return a canonical HexStrike tool result while preserving existing keys."""
    if not isinstance(result, dict):
        result = {"success": False, "error": f"Invalid tool result type: {type(result).__name__}"}

    normalized = dict(result)
    success = bool(normalized.get("success", False))
    output = normalized.get("output")
    error = normalized.get("error")

    if output is None:
        output = normalized.get("stdout", "")
    if error is None:
        error = normalized.get("stderr", "") if not success else ""

    normalized["success"] = success
    normalized["output"] = "" if output is None else output
    normalized["error"] = "" if error is None else error
    normalized["returncode"] = normalized.get("returncode", normalized.get("return_code", -1))
    normalized["timed_out"] = bool(normalized.get("timed_out", False))
    normalized["partial_results"] = bool(normalized.get("partial_results", False))
    normalized["execution_time"] = normalized.get("execution_time", 0.0)
    normalized["timestamp"] = normalized.get("timestamp", "")
    return normalized


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


def _suggest_next_tool(tool_name: str, output: str, target: str = "") -> dict:
    """Suggest the next tool based on current tool output.

    Returns dict with 'tool' (str) and 'reason' (str), or empty dict if context is insufficient.
    Used by run_security_tool (exec tools) and pulse_app tools.
    """
    output_lower = output.lower()

    # Port scanning — detect web services
    if tool_name in ("nmap", "nmap_advanced", "masscan", "rustscan"):
        if "80/" in output_lower or "443/" in output_lower or "http" in output_lower:
            return {"tool": "whatweb", "reason": "Web ports detected — identify technologies"}
        if "445/" in output_lower:
            return {"tool": "smbmap", "reason": "SMB port 445 open — enumerate shares"}
        if "22/" in output_lower:
            return {"tool": "hydra", "reason": "SSH port 22 open — test credentials"}
        if "1433/" in output_lower or "3306/" in output_lower:
            return {"tool": "sqlmap", "reason": "Database port open — test for weak auth"}
        return {"tool": "nuclei", "reason": "Ports discovered — run vulnerability scan"}

    # Web fingerprinting — detect CMS
    if tool_name == "whatweb":
        if "wordpress" in output_lower:
            return {"tool": "wpscan", "reason": "WordPress detected — enumerate plugins/users"}
        if "joomla" in output_lower:
            return {"tool": "joomscan", "reason": "Joomla detected — enumerate extensions"}
        if "drupal" in output_lower:
            return {"tool": "nuclei", "reason": "Drupal detected — check known CVEs"}
        if output.strip():
            return {"tool": "gobuster", "reason": "Web server detected — discover hidden paths"}

    # Vulnerability scan results
    if tool_name in ("nuclei", "nikto"):
        if "sql" in output_lower or "sqli" in output_lower or "injection" in output_lower:
            return {"tool": "sqlmap", "reason": "SQL injection candidate found — confirm and exploit"}
        if "xss" in output_lower or "cross-site" in output_lower:
            return {"tool": "dalfox", "reason": "XSS candidate found — validate with dalfox"}
        if "ssl" in output_lower or "tls" in output_lower or "certificate" in output_lower:
            return {"tool": "testssl", "reason": "SSL/TLS issues reported — deep inspection"}
        if "smb" in output_lower or "eternalblue" in output_lower or "ms17" in output_lower:
            return {"tool": "metasploit", "reason": "SMB vulnerability confirmed — attempt exploitation"}
        if output.strip():
            return {"tool": "gobuster", "reason": "Vuln scan complete — continue with directory discovery"}

    # Directory enumeration
    if tool_name in ("gobuster", "ffuf", "dirsearch", "feroxbuster"):
        if output.strip():
            return {"tool": "nuclei", "reason": "Directories discovered — scan for vulnerabilities"}

    # Password cracking
    if tool_name in ("hydra", "medusa", "patator"):
        if "success" in output_lower or "password" in output_lower:
            return {"tool": "metasploit", "reason": "Credentials found — attempt exploitation"}

    # SMB enumeration
    if tool_name in ("smbmap", "enum4linux", "nbtscan", "netexec"):
        if "share" in output_lower or "admin" in output_lower or "ipc" in output_lower:
            return {"tool": "metasploit", "reason": "SMB shares accessible — check for exploitation"}
        return {"tool": "hydra", "reason": "SMB services detected — test credentials"}

    # Web vulnerability tools
    if tool_name in ("sqlmap", "dalfox", "xsser"):
        if "vulnerable" in output_lower or "payload" in output_lower or "parameter" in output_lower:
            return {"tool": "metasploit", "reason": "Web vulnerability confirmed — attempt exploitation"}
        return {"tool": "nuclei", "reason": "Web check complete — broader vulnerability scan"}

    return {}


_TOOL_COUCHE1: Dict[str, Dict[str, str]] = {
    "nmap": {
        "workflow": "FIRST recon tool on any new target. Run before whatweb.",
        "example": "nmap(target='scanme.nmap.org') or nmap(target='192.168.1.1', ports='80,443')",
        "returns": [
            "dict — success (bool), output (str) with port lines like '22/tcp open  ssh'",
            "Parse output line by line: split on whitespace, port/protocol in first column, state in second, service in third.",
        ],
    },
    "whatweb": {
        "workflow": "Web technology detection. Use AFTER nmap when web ports are found.",
        "example": "whatweb(url='http://scanme.nmap.org')",
        "returns": [
            "dict — success (bool), output (str) with '[200 OK] nginx PHP' tech fingerprinting results.",
            "Check output for technology keywords: nginx, Apache, PHP, Python, WordPress, etc.",
        ],
    },
    "sqlmap": {
        "workflow": "SQL injection testing. Use AFTER finding SQLi candidates via findings.",
        "example": "sqlmap(url='http://target/page?id=1') or sqlmap(url='http://target/page?id=1', additional_args='--batch')",
        "returns": [
            "dict — success (bool), output (str) with injection details and dumped data.",
            "Filter output for 'Parameter:' and 'Type:' lines to identify injectable parameters.",
        ],
    },
    "gobuster": {
        "workflow": "Directory/file brute force. Use AFTER whatweb when web server is detected.",
        "example": "gobuster(url='http://target', mode='dir')",
        "returns": [
            "dict — success (bool), output (str) with discovered paths like '/admin (Status: 200)'.",
            "Filter output for '(Status: 2..)' lines to find accessible paths.",
        ],
    },
    "nuclei": {
        "workflow": "Vulnerability scanning. Use AFTER surface scan for known CVEs.",
        "example": "nuclei(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with findings like '[medium] [missing-header] http://target'.",
            "Filter output for lines matching '[severity] [template-id] url' — severity is one of: critical, high, medium, low, info.",
        ],
    },
    "nikto": {
        "workflow": "Web server vulnerability scanner. Use AFTER whatweb.",
        "example": "nikto(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with findings like '+ /admin: Admin login page'.",
            "Filter output for '+ ' prefixed lines to find issues.",
        ],
    },
    "masscan": {
        "workflow": "Fast port scanner for large ranges. Use BEFORE or INSTEAD OF nmap on /24+ subnets.",
        "example": "masscan(target='10.10.10.0/24', ports='1-65535', rate=1000)",
        "returns": [
            "dict — success (bool), output (str) with discovered hosts and ports (masscan format).",
            "Parse output for 'Discovered open port' lines to extract port/protocol/host.",
        ],
    },
    "rustscan": {
        "workflow": "Ultra-fast port scanner. Use INSTEAD OF nmap when speed matters on single hosts.",
        "example": "rustscan(target='10.10.10.1')",
        "returns": [
            "dict — success (bool), output (str) with nmap-compatible port scan results.",
            "Output format mirrors nmap: '22/tcp open  ssh' lines in the results.",
        ],
    },
    "subfinder": {
        "workflow": "FIRST recon on any new domain target. Run before fierce or httpx.",
        "example": "subfinder(domain='example.com')",
        "returns": [
            "dict — success (bool), output (str) with subdomains, one per line.",
            "Filter output for lines containing the target domain to find valid subdomains.",
        ],
    },
    "fierce": {
        "workflow": "DNS reconnaissance. Use AFTER subfinder for deeper DNS enumeration.",
        "example": "fierce(domain='example.com')",
        "returns": [
            "dict — success (bool), output (str) with DNS records, zone transfers, hostnames.",
            "Check output for 'Found: ' lines to identify live hosts via DNS.",
        ],
    },
    "dirsearch": {
        "workflow": "Web path enumeration. Use AFTER whatweb when web server is detected.",
        "example": "dirsearch(url='http://target')",
        "returns": [
            "dict — success (bool), output (str) with discovered paths and status codes.",
            "Filter output for '200' or '301' status codes to find accessible paths.",
        ],
    },
    "feroxbuster": {
        "workflow": "Recursive content discovery. Use AFTER whatweb for deep crawling.",
        "example": "feroxbuster(url='http://target')",
        "returns": [
            "dict — success (bool), output (str) with recursively discovered paths and status codes.",
            "Filter output for successful status codes (200, 301, 403) to find paths.",
        ],
    },
    "dalfox": {
        "workflow": "XSS vulnerability scanning. Use AFTER whatweb when reflected params are found.",
        "example": "dalfox(url='http://target/page?param=value')",
        "returns": [
            "dict — success (bool), output (str) with XSS vulnerability findings and PoC URLs.",
            "Filter output for '[V]' or 'Vulnerability' lines to find confirmed XSS vectors.",
        ],
    },
    "xsser": {
        "workflow": "Cross-site scripting testing. Use AFTER whatweb for reflected XSS analysis.",
        "example": "xsser(url='http://target/page?param=value')",
        "returns": [
            "dict — success (bool), output (str) with XSS testing results and payload reflections.",
            "Check output for 'Vulnerable' keyword to identify injectable parameters.",
        ],
    },
    "joomscan": {
        "workflow": "Joomla vulnerability scanner. Use AFTER whatweb when Joomla CMS is detected.",
        "example": "joomscan(url='http://target')",
        "returns": [
            "dict — success (bool), output (str) with Joomla version, extensions, vulnerabilities.",
            "Filter output for '[+]' lines to find extensions and '[!]' for vulnerabilities.",
        ],
    },
    "dotdotpwn": {
        "workflow": "Directory traversal scanner. Use AFTER whatweb when web server is detected.",
        "example": "dotdotpwn(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with traversal test results and vulnerable paths.",
            "Check output for 'Vulnerable' or 'Found' lines to identify path traversal bugs.",
        ],
    },
    "smbmap": {
        "workflow": "SMB share enumeration. Use AFTER nmap when SMB port 445 is open.",
        "example": "smbmap(target='10.10.10.1')",
        "returns": [
            "dict — success (bool), output (str) with SMB shares, permissions, and accessible files.",
            "Check output for 'Disk' shares with read/write access — focus on ADMIN$, IPC$.",
        ],
    },
    "hydra": {
        "workflow": "Network login brute-forcer. Use AFTER nmap when services need password testing.",
        "example": "hydra(target='10.10.10.1', service='ssh')",
        "returns": [
            "dict — success (bool), output (str) with login attempts and cracked credentials.",
            "Filter output for 'login:' and 'password:' lines to find successful logins.",
        ],
    },
    "medusa": {
        "workflow": "Network login brute-forcer. Use AFTER nmap for parallel password testing.",
        "example": "medusa(target='10.10.10.1', module='ssh')",
        "returns": [
            "dict — success (bool), output (str) with login attempts and discovered credentials.",
            "Filter output for 'ACCOUNT FOUND' lines to identify successful authentication.",
        ],
    },
    "patator": {
        "workflow": "Multi-purpose brute-forcer. Use AFTER nmap for custom protocol brute-force.",
        "example": "patator(target='10.10.10.1', module='ftp_login')",
        "returns": [
            "dict — success (bool), output (str) with brute-force attempt results per target.",
            "Check output for matches or successful authentication indicators per module.",
        ],
    },
    "prowler": {
        "workflow": "Cloud security audit. Run standalone against your cloud provider.",
        "example": "prowler(provider='aws')",
        "returns": [
            "dict — success (bool), output (str) with cloud security findings and compliance checks.",
            "Filter output for 'FAIL' lines to find misconfigurations and security issues.",
        ],
    },
    "msfvenom": {
        "workflow": "Metasploit payload generator. Use AFTER metasploit when custom payloads are needed.",
        "example": "msfvenom(payload='linux/x64/shell_reverse_tcp', lhost='10.0.0.1')",
        "returns": [
            "dict — success (bool), output (str) with payload generation status and file path.",
            "The generated binary/payload is written to the output file specified in params.",
        ],
    },
    "ropgadget": {
        "workflow": "ROP gadget finder. Use AFTER checksec when binary exploitation is needed.",
        "example": "ropgadget(file='/path/to/binary')",
        "returns": [
            "dict — success (bool), output (str) with ROP gadgets, one per line, sorted by address.",
            "Filter output for '0x' addresses with useful gadgets: 'pop rdi', 'syscall', 'ret'.",
        ],
    },
    "wpscan": {
        "workflow": "WordPress vulnerability scanner. Use AFTER whatweb when WordPress CMS is detected.",
        "example": "wpscan(url='http://target')",
        "returns": [
            "dict — success (bool), output (str) with vulnerabilities, user enumeration, plugin versions.",
            "Filter output for '[!]' lines to find vulnerabilities; '[i]' for informational findings.",
        ],
    },
    "httpx": {
        "workflow": "HTTP probing and tech detection. Use AFTER subfinder on domain targets.",
        "example": "httpx(target='example.com')",
        "returns": [
            "dict — success (bool), output (str) with HTTP status, title, tech stack per endpoint.",
            "Filter output for status codes 200/301/403 to identify live endpoints.",
        ],
    },
    "amass": {
        "workflow": "Subdomain enumeration and OSINT. Use FIRST on domain targets before subfinder.",
        "example": "amass(domain='example.com')",
        "returns": [
            "dict — success (bool), output (str) with discovered subdomains, one per line.",
            "Filter output for lines containing the target domain to find subdomains.",
        ],
    },
    "waybackurls": {
        "workflow": "Historical URL discovery. Use AFTER subfinder for endpoint discovery.",
        "example": "waybackurls(target='example.com')",
        "returns": [
            "dict — success (bool), output (str) with historical URLs, one per line.",
            "Filter output for interesting extensions: .php, .asp, .js, .json, .config.",
        ],
    },
    "metasploit": {
        "workflow": "Exploitation framework. Use AFTER plan_attack or findings when exploit modules are identified.",
        "example": "metasploit(module='exploit/multi/handler', target='10.10.10.1')",
        "returns": [
            "dict — success (bool), output (str) with exploit execution results.",
            "Check for 'Meterpreter session' or 'Command shell session' lines for successful exploitation.",
        ],
    },
    "impacket": {
        "workflow": "AD/Windows exploitation toolkit. Use AFTER enum4linux or smbmap when SMB/AD services found.",
        "example": "impacket(script='secretsdump', target='10.10.10.1')",
        "returns": [
            "dict — success (bool), output (str) with script execution results.",
            "Impacket scripts: GetADUsers, secretsdump, smbclient, psexec, wmiexec.",
        ],
    },
    "wafw00f": {
        "workflow": "WAF detection. Use BEFORE whatweb when web server is suspected to be behind WAF.",
        "example": "wafw00f(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with WAF name if detected, or 'No WAF detected'.",
            "Check output for known WAF names: Cloudflare, Cloudfront, AWS WAF, ModSecurity.",
        ],
    },
    "hashcat": {
        "workflow": "Password hash cracking. Use AFTER hashid or when hash files are obtained.",
        "example": "hashcat(hash_file='hashes.txt', mode='0', wordlist='/usr/share/wordlists/rockyou.txt')",
        "returns": [
            "dict — success (bool), output (str) with cracked passwords and status.",
            "Filter output for lines matching 'hash:password' format to find cracked credentials.",
        ],
    },
    "whois": {
        "workflow": "Domain registration lookup. Use FIRST on any new domain target for OSINT.",
        "example": "whois(target='example.com')",
        "returns": [
            "dict — success (bool), output (str) with registrar, creation date, name servers.",
            "Check registrant email and organization for organizational OSINT.",
        ],
    },
    "ffuf": {
        "workflow": "Fast web fuzzer. Use AFTER gobuster for more granular directory discovery.",
        "example": "ffuf(url='http://target/FUZZ', wordlist='/usr/share/wordlists/dirb/common.txt')",
        "returns": [
            "dict — success (bool), output (str) with discovered paths and HTTP status codes.",
            "Filter output for 'Status: 200' or 'Status: 301' to find accessible paths.",
        ],
    },
    "zap": {
        "workflow": "OWASP ZAP web scanner. Use AFTER whatweb for active web vulnerability scanning.",
        "example": "zap(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with active scan alerts and findings.",
            "Check output for alert severity: High, Medium, Low, Informational.",
        ],
    },
    "jaeles": {
        "workflow": "Web security scanning framework. Use AFTER surface scan for signature-based detection.",
        "example": "jaeles(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with detected vulnerabilities from YAML signatures.",
            "Filter output for 'Vulnerability Found' or severity tags.",
        ],
    },
    "gau": {
        "workflow": "URL gathering from public sources. Use AFTER waybackurls for additional endpoint discovery.",
        "example": "gau(target='example.com')",
        "returns": [
            "dict — success (bool), output (str) with gathered URLs from AlienVault and Wayback.",
            "Filter output for specific file extensions (.php, .js) or parameters (?, =).",
        ],
    },
    "katana": {
        "workflow": "Web crawler for endpoint discovery. Use AFTER httpx for deep crawling of live endpoints.",
        "example": "katana(target='http://target')",
        "returns": [
            "dict — success (bool), output (str) with crawled endpoints and JavaScript sources.",
            "Filter output for endpoints with query parameters or interesting paths.",
        ],
    },
    "http_request": {
        "workflow": "Generic HTTP client. Use for GET/POST/PUT/DELETE requests, form login, API probing, cookie capture. Returns structured response with status_code, headers, cookies, and body.",
        "example": "http_request(method='GET', url='http://target/login.php') then http_request(method='POST', url='http://target/login.php', data='username=admin&password=admin')",
        "returns": [
            "dict — success (bool), status_code (int), ok (bool, 2xx), headers (dict), cookies (dict), body (str), body_truncated (bool).",
            "Cookies from Set-Cookie are parsed into the cookies dict automatically.",
        ],
    },
    "raw_tcp": {
        "workflow": "Raw TCP socket client. Use for protocol-level attacks when http_request cannot send non-HTTP payloads. Payload is hex-encoded to avoid JSON encoding issues.",
        "example": "raw_tcp(host='10.10.10.1', port=1337, payload_hex='474554202f20485454502f312e310d0a0d0a') sends GET / HTTP/1.1",
        "returns": [
            "dict — success (bool), response_hex (str, hex-encoded response), bytes_sent (int), bytes_recv (int), duration_ms (int).",
            "Decode response_hex with bytes.fromhex(response_hex).decode(errors='replace') to read as text.",
        ],
    },
    "execute_code": {
        "workflow": "Execute arbitrary code. Use for inline scripting, crypto/math (BLS, ECC, P-256, modular arithmetic, LLL lattice), one-liner exploits, payload encoding/decoding. Supported languages: python, bash, node.",
        "example": "execute_code(code='import hashlib; print(hashlib.md5(b\"test\").hexdigest())', language='python')",
        "returns": [
            "dict — success (bool), stdout (str), stderr (str), exit_code (int), language (str), timeout (bool).",
            "Check exit_code == 0 for success. stderr may contain warnings even on success.",
        ],
    },
    "browser_fetch": {
        "workflow": "Headless browser page fetch. Use when curl/http_request returns empty/missing content due to JS rendering. Requires Playwright installed.",
        "example": "browser_fetch(url='http://target/spa-page') then browser_eval(url='http://target', js='document.querySelector(\\'#token\\').innerText')",
        "returns": [
            "dict — success (bool), html (str, post-JS rendered DOM), title (str), url (str).",
            "Returns full HTML after JS execution — compare with http_request to detect SPA/client-side rendering.",
        ],
    },
    "browser_screenshot": {
        "workflow": "Headless browser screenshot. Use for visual verification of web pages, CAPTCHA reading, or dashboard inspection. Requires Playwright installed.",
        "example": "browser_screenshot(url='http://target') returns base64-encoded PNG of the full page.",
        "returns": [
            "dict — success (bool), screenshot_base64 (str, base64-encoded PNG), title (str), url (str).",
            "Pass selector parameter to screenshot only a specific element (e.g. '#main-content').",
        ],
    },
    "browser_eval": {
        "workflow": "Execute JavaScript in headless browser context. Use for extracting values from JS-rendered pages, reading localStorage/cookies, or manipulating DOM for XSS testing. Requires Playwright installed.",
        "example": "browser_eval(url='http://target/admin', js='document.cookie') extracts cookies after login.",
        "returns": [
            "dict — success (bool), result (str, stringified JS return value), title (str), url (str).",
            "The return value of the JS expression is converted to string — use JSON.stringify() for objects.",
        ],
    },
    "pwntools": {
        "workflow": "CTF binary exploitation (pwn/rev). Write pwntools exploit script in script_content. Use AFTER file/checksec/objdump analysis. Supports ret2libc, ret2dlresolve, ROP chains, format string, shellcode injection. Connects to remote target via target_host:target_port or local via target_binary.",
        "example": "Run pwntools(script_content='from pwn import *; e = ELF(\"./void\"); io = remote(\"host\", 1337); io.send(payload); io.interactive()', target_host='10.0.0.1', target_port=1337)",
        "returns": [
            "dict — success (bool), output (str) with pwntools script stdout, stderr, and timing.",
            "The output contains exploit result — flag for CTF, shell output, or error message.",
            "Check output for flag patterns like HTB{...}, flag{...}, CTF{...}.",
        ],
    },
}


def _build_typed_tool_doc(tool_name: str, description: str, tool_def: Dict[str, Any]) -> str:
    """Generate a rich docstring for a typed wrapper tool (Couche 1 plug-and-play)."""
    lower = tool_name.lower()
    couche1 = _TOOL_COUCHE1.get(lower, {})

    lines = [description, ""]

    if "workflow" in couche1:
        lines.append(f"Workflow: {couche1['workflow']}")
        lines.append("")

    lines.append("Args:")
    for param_name, spec in tool_def.get("params", {}).items():
        lines.append(f"    {param_name}: Required")
    for param_name, default in tool_def.get("optional", {}).items():
        lines.append(f"    {param_name}: Optional. Default: {default!r}")

    lines.append("")
    if "returns" in couche1:
        lines.append("Returns:")
        for ret_line in couche1["returns"]:
            lines.append(f"    {ret_line}")
    else:
        lines.append("Returns:")
        lines.append("    dict with success (bool), output (str), execution_time (float), error (str)")
        lines.append("    Fields: success, output, error, returncode, timed_out, execution_time")

    if "example" in couche1:
        lines.append("")
        lines.append("Example:")
        lines.append(f"    {couche1['example']}")

    lines.append("")
    lines.append("Note:")
    lines.append(f"    Typed wrapper over run_security_tool(tool_name={tool_name!r})")
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
            supporting_files="template",
            reload=True,
        )
    )
    logger.info("🤖 Skills initialized (main files visible, supporting files via template, reload enabled)")

async def run_security_tool(
    ctx: Context,
    tool_name: str,
    parameters: str | Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute any security tool from the HexStrike arsenal.

    Args:
        tool_name:  Tool name (e.g. 'nmap', 'sqlmap', 'sherlock')
        parameters: JSON string or dict of parameters (e.g. '{"target": "example.com"}')

    Returns:
        Tool execution results
    """
    _t_start = time.time()
    _request_id = get_request_id()
    _telemetry: Dict[str, Any] = {
        "tool":             tool_name,
        "success":          False,
        "duration":         0.0,
        "timed_out":        False,
        "cache_hit":        False,
        "session_state":    False,
        "confirmation":     None,
        "opt_profile":      "normal",
        "skill_injected":   False,
        "prompt_suggested": False,
        "target":           "",
        "session_id":       "",
        "request_id":       _request_id or "",
    }
    await ctx.info(f"🔍 Executing {tool_name}")

    def finalize(result: Any) -> Dict[str, Any]:
        normalized = _normalize_tool_result(result)
        _telemetry["success"] = bool(normalized.get("success", False))
        _telemetry["timed_out"] = bool(normalized.get("timed_out", False))
        _telemetry["duration"] = round(time.time() - _t_start, 3)
        logger.info("[telemetry] %s", json.dumps(_telemetry))
        _op_metrics.record(_telemetry)
        _telemetry["event"] = "tool_execution"
        _pipeline.emit(_telemetry)
        get_tool_stats_store().record(
            tool_name,
            success=_telemetry["success"],
        )
        suggestion = _suggest_next_tool(
            tool_name,
            normalized.get("output", ""),
            _telemetry.get("target", ""),
        )
        if suggestion:
            normalized["next_suggested_tool"] = suggestion
        return normalized

    try:
        params = json.loads(parameters) if isinstance(parameters, str) else parameters
    except json.JSONDecodeError as e:
        await ctx.error(f"❌ Invalid JSON parameters: {e}")
        return finalize({"success": False, "error": f"Invalid JSON: {e}"})

    if not isinstance(params, dict):
        await ctx.error("❌ Invalid parameters: expected JSON object")
        return finalize({"success": False, "error": "Invalid parameters: expected JSON object"})

    route = _DIRECT_TOOLS_CACHE.get(tool_name.lower())
    if not route:
        await ctx.error(f"❌ Unknown tool: {tool_name}")
        return finalize({"success": False, "error": f"Unknown tool: {tool_name}"})

    mod_path, func_name, tool_key = route
    exec_func = _resolve_exec_func(mod_path, func_name)
    if exec_func is None:
        await ctx.error(f"❌ Executor not found: {func_name} in {mod_path}")
        return finalize({"success": False, "error": f"Executor {func_name} not found in {mod_path}"})

    target = (
        params.get("target")
        or params.get("url")
        or params.get("domain")
        or params.get("interface", "")
    )
    _telemetry["target"] = target or ""

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
            return finalize({
                "success": False,
                "error": f"Action cancelled - {destructive_request['action']} requires explicit confirmation",
            })
        _telemetry["confirmation"] = "accepted"
    else:
        _telemetry["confirmation"] = "skipped"

    skill_name = _TOOL_SKILL_MAP.get(tool_name.lower())
    if skill_name:
        skill_doc = await _read_skill_document(ctx, skill_name, "SKILL.md")
        if skill_doc:
            lines = [l for l in skill_doc.splitlines() if l.strip() and not l.startswith("---")]
            header = "\n".join(lines[:4])
            await ctx.info(f"📚 [{skill_name}] {header}")
            _telemetry["skill_injected"] = True

    opt_profile = params.pop("_profile", "normal")
    tech_dict   = params.pop("_tech", None)
    tech_profile: Optional[TechProfile] = None

    if isinstance(tech_dict, dict):
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
            logger.debug("Failed to build TechProfile from whatweb output", exc_info=True)
    else:
        if target:
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
                logger.debug("Failed to restore tech profile from session state", exc_info=True)

            if tech_profile is None:
                tech_profile = _detect_from_cache(target)
                if tech_profile:
                    await ctx.info(f"🧠 Tech detected from cache: {tech_profile.summary()}")
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
                        logger.debug("Failed to persist tech profile to session state", exc_info=True)

    if tech_profile:
        suggested_prompt = None
        if tech_profile.cms and any(c in tech_profile.cms for c in ("wordpress", "joomla", "drupal")):
            suggested_prompt = ("bug_bounty_recon", {"target": target or tool_name})
        elif tech_profile.security and any(s in tech_profile.security for s in ("cloudflare", "akamai", "aws_waf")):
            suggested_prompt = ("bug_bounty_recon", {"target": target or tool_name})
        elif "smb" in (tech_profile.services or []) or "ldap" in (tech_profile.services or []):
            suggested_prompt = ("smb_lateral_movement", {"target": target or "unknown"})
        elif any(s in (tech_profile.services or []) for s in ("ssh", "rdp")) and not target:
            pass

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
                logger.debug("Failed to suggest workflow prompt", exc_info=True)

    if not opt_profile or opt_profile == "normal":
        try:
            saved_rl = await ctx.get_state(f"ratelimit:{target}")
            if saved_rl:
                opt_profile = saved_rl
                _telemetry["opt_profile"] = opt_profile
                await ctx.info(f"⚡ Rate limit profile restored: {opt_profile}")
        except Exception:
            logger.debug("Failed to restore rate limit profile from session state", exc_info=True)

    _telemetry["opt_profile"] = opt_profile

    _session_id = ctx.session_id
    _cache_key = _cache_key_for(_session_id, tool_name, target, params) if target else None
    if _cache_key:
        prior = _scan_cache.get(_cache_key)
        if prior and prior.get("result", {}).get("success"):
            _telemetry["cache_hit"] = True
            _telemetry["success"] = True
            await ctx.info(f"⚡ {tool_name} cache hit for {target} — returning cached result")
            return finalize(prior["result"])
        prior = _scan_cache.get(f"seed:{tool_name}:{target}")
        if prior and prior.get("result", {}).get("success"):
            _telemetry["cache_hit"] = True
            _telemetry["success"] = True
            await ctx.info(f"⚡ {tool_name} seed cache hit for {target}")
            return finalize(prior["result"])

    params = _optimizer.optimize(tool_name.lower(), params, tech_profile, opt_profile)
    optimizer_meta = params.pop("_optimizer", {})
    if optimizer_meta.get("forced_stealth"):
        await ctx.info(f"🛡️ WAF detected → stealth mode forced for {tool_name}")
    elif opt_profile != "normal":
        await ctx.info(f"⚙️ Profile: {optimizer_meta.get('profile', opt_profile)}")

    loop = asyncio.get_running_loop()
    future = asyncio.ensure_future(
        loop.run_in_executor(None, lambda: exec_func(tool_name.lower(), params))
    )

    try:
        await ctx.report_progress(0, 100)
        phases = [(25, "🔧 Preparing..."), (50, "⚙️  Running..."), (75, "📊 Processing results...")]
        tick = 12
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=tick)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        pct = 80
        while not future.done():
            done, _ = await asyncio.wait([future], timeout=15)
            if done:
                break
            await ctx.report_progress(pct, 100)
            await ctx.info(f"⏳ Still running ({pct}%)")
            pct = min(pct + 5, 98)

        result = _normalize_tool_result(await future)
        await ctx.report_progress(100, 100)
    except asyncio.CancelledError:
        result = {"success": False, "error": "Tool execution timed out", "timed_out": True}
        await ctx.report_progress(100, 100)
    except Exception as exc:
        result = {"success": False, "error": str(exc)[:500], "timed_out": False}

    _telemetry["timed_out"] = result.get("timed_out", False)

    if result.get("success"):
        await ctx.info(f"✅ {tool_name} completed")
        _telemetry["success"] = True

        output = str(result.get("output", "") or result.get("data", ""))
        status_code = result.get("returncode", 0) or 0
        rl = _rate_limiter.detect_rate_limiting(output, status_code)
        if rl["detected"]:
            recommended = rl["recommended_profile"]
            _telemetry["rate_limit"] = recommended
            _rate_limit_events.append({
                "target":    target or "",
                "tool":      tool_name,
                "profile":   recommended,
                "confidence": rl["confidence"],
                "indicators": rl.get("indicators", []),
                "timestamp":  time.time(),
            })
            await ctx.info(f"⚠️ Rate limit detected (confidence={rl['confidence']:.0%}) → switching to {recommended}")
            try:
                await ctx.set_state(f"ratelimit:{target}", recommended)
            except Exception:
                logger.debug("Failed to persist rate limit profile to session state", exc_info=True)

        if target:
            exec_time = result.get("execution_time", 0.0)
            _scan_cache.set(_cache_key or _cache_key_for(_session_id, tool_name, target or "unknown", params), {
                "tool": tool_name, "target": target,
                "result": result, "timestamp": time.time(),
            }, execution_time=exec_time)
            _telemetry["cache_hit"] = False
    else:
        await ctx.error(f"❌ {tool_name} failed: {result.get('error', 'unknown')}")
        error_msg = str(result.get("error", "") or result.get("stderr", "") or "")
        if error_msg:
            from server_core.singletons import error_handler as _eh
            error_type = _eh.classify_error(error_msg)
            result["error_type"] = error_type.value
            _telemetry["error_type"] = error_type.value
            _eh.handle_tool_failure(
                tool_name,
                Exception(error_msg[:500]),
                {"target": target or "unknown", "parameters": params},
            )
            alternative = _eh.get_alternative_tool(tool_name, {})
            if alternative:
                result["suggested_alternative"] = alternative
                await ctx.info(f"💡 Try: {alternative} (error: {error_type.value})")

    if params.get("_ai_suggest") and result.get("success"):
        output_text = str(result.get("output", ""))[:2000]
        if output_text.strip():
            try:
                sample_result = await ctx.sample(
                    messages=(
                        f"You are a penetration testing assistant analyzing tool output.\n"
                        f"Tool: {tool_name}\nTarget: {target or 'unknown'}\n"
                        f"Output (truncated to 2000 chars):\n{output_text}\n\n"
                        f"Based on this output, suggest the single most valuable next "
                        f"HexStrike tool to run and why. Be concise (2-3 sentences max). "
                        f"Format: 'Next tool: <tool_name> — <reason>'"
                    ),
                    max_tokens=120,
                )
                suggestion = sample_result.text.strip() if sample_result else ""
                if suggestion:
                    await ctx.info(f"🤖 AI suggestion: {suggestion}")
                    result["ai_suggestion"] = suggestion
                    _telemetry["ai_suggested"] = True
            except Exception:
                logger.debug("ctx.sample not supported by client or failed", exc_info=True)

    return finalize(result)


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

    # RegexSearchTransform: collapse all tools into search_tools + call_tool to save context.
    # markdown serializer uses ~40% fewer tokens than JSON; always_visible pins essential tools.
    transforms = [RegexSearchTransform(
        max_results=10,
        search_result_serializer=serialize_tools_for_output_markdown,
        always_visible=["scan", "get_live_dashboard", "scan_background", "pulse_dashboard"],
    )] if RegexSearchTransform else []
    from mcp_core.instructions import INSTRUCTIONS
    mcp = FastMCP(
        "hexstrike-ai pulse",
        instructions=os.environ.get("HEXSTRIKE_INSTRUCTIONS", INSTRUCTIONS),
        transforms=transforms,
    )

    # Middleware — framework-level logging and session tracking
    mcp.add_middleware(HexStrikeSessionMiddleware())
    mcp.add_middleware(HexStrikeLoggingMiddleware(
        log_resources=False,   # enable when debugging resource access
        log_prompts=True,
    ))

    _register_skills(mcp, logger)

    # Populate DIRECT_TOOLS cache from shared TOOL_ROUTES
    _populate_direct_tools()
    DIRECT_TOOLS = _DIRECT_TOOLS_CACHE

    # run_security_tool is now defined at module level (above setup_mcp_server_standalone)
    # It is NOT exposed as an @mcp.tool() — agents must use typed tools or scan/ctf entry points

    @mcp.tool(
        description="Return the local skill bundle associated with a HexStrike tool",
        annotations={"readOnlyHint": True, "openWorldHint": False},
        task=True,
        timeout=None,
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
        task=True,
        timeout=None,
    )
    async def plan_attack(
        ctx: Context,
        target: str,
        objective: str = "comprehensive",
        ctf_category: str = "",
        ctf_difficulty: str = "unknown",
        ctf_points: int = 0,
        ctf_description: str = "",
    ) -> Dict[str, Any]:
        """
        Generate an intelligent attack chain for a target.

        Args:
            target:    Target (IP, URL, domain, file path)
            objective: comprehensive | quick | stealth | ctf | bug_bounty_recon |
                       bug_bounty_hunting | aws | kubernetes | containers | iac
            ctf_category:   CTF challenge category (web|crypto|pwn|forensics|rev|misc|osint)
            ctf_difficulty: CTF difficulty (easy|medium|hard|insane)
            ctf_points:     CTF point value
            ctf_description: CTF challenge description

        Returns:
            AttackChain dict with ordered steps, tools, probabilities, estimated time
        """
        await ctx.info(f"🧠 Analyzing target: {target} (objective={objective})")
        await ctx.report_progress(0, 100)

        # -----------------------------------------------------------------------
        # CTF objective — use CTFWorkflowManager for rich per-category workflows
        # -----------------------------------------------------------------------
        if objective == "ctf":
            from server_core.workflows.ctf.CTFChallenge import CTFChallenge
            from server_core.workflows.ctf.workflowManager import CTFWorkflowManager
            from shared.target_profile import TargetProfile
            from shared.attack_chain import AttackChain
            from shared.attack_step import AttackStep
            from shared.target_types import TargetType

            if not ctf_category:
                ctf_category = "web"
            target_str = str(target)
            points_str = f" | {ctf_points}pts" if ctf_points else ""
            await ctx.info(f"🏴 CTF Workflow — [{ctf_category.upper()}] {target_str} ({ctf_difficulty}{points_str})")

            challenge = CTFChallenge(
                name=target_str,
                category=ctf_category,
                description=ctf_description or f"CTF challenge targeting {target_str}",
                difficulty=ctf_difficulty,
                points=ctf_points,
                target=target_str,
            )

            ctf_wfm = CTFWorkflowManager()
            workflow = ctf_wfm.create_ctf_challenge_workflow(challenge)

            profile = TargetProfile(target=target_str, target_type=TargetType.CTF_CHALLENGE)
            profile.risk_level = "medium" if ctf_difficulty in ("hard", "insane") else "low"
            chain = AttackChain(profile)
            chain.risk_level = profile.risk_level

            manual_tools = {"manual", "custom", "python", "sage", "ida",
                            "x64dbg", "ollydbg", "burpsuite", "wireshark",
                            "audacity", "maltego"}

            for step in workflow.get("workflow_steps", []):
                tools = step.get("tools", ["manual"])
                step_desc = step.get("description", f"CTF step: {step.get('action', 'unknown')}")
                step_time = step.get("estimated_time", 600)
                for tool in tools:
                    if tool not in manual_tools:
                        step_obj = AttackStep(
                            tool=tool,
                            parameters={"target": target_str, "category": ctf_category},
                            expected_outcome=step_desc,
                            success_probability=workflow.get("success_probability", 0.5),
                            execution_time_estimate=step_time,
                        )
                        chain.add_step(step_obj)

            chain.calculate_success_probability()

            result = chain.to_dict()
            result["ctf_metadata"] = {
                "category": ctf_category,
                "difficulty": ctf_difficulty,
                "points": ctf_points,
                "strategies": workflow.get("strategies", []),
                "parallel_tasks": workflow.get("parallel_tasks", []),
                "automation_level": workflow.get("automation_level", "high"),
                "resource_requirements": workflow.get("resource_requirements", {}),
                "validation_steps": workflow.get("validation_steps", []),
            }

            await ctx.info(
                f"✅ CTF attack chain ready: {len(chain.steps)} steps | "
                f"{len(workflow.get('strategies', []))} strategies | "
                f"success: {chain.success_probability:.0%}"
            )
            await ctx.report_progress(100, 100)
            return result

        # -----------------------------------------------------------------------
        # Standard objective — use IntelligentDecisionEngine
        # -----------------------------------------------------------------------
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
            logger.debug("Failed to restore target profile from session", exc_info=True)

        if profile is None:
            profile = await loop.run_in_executor(None, lambda: _ide.analyze_target(target))
            await ctx.report_progress(50, 100)
            # Enrich profile with cached scan results before creating the attack chain
            cached_scans = _collect_cached_scans(ctx.session_id, target)
            if cached_scans:
                profile = _enrich_profile_from_cache(profile, cached_scans)
                await ctx.info(f"📦 Injected {len(cached_scans)} cached scan(s) into profile")
            await ctx.info(f"🎯 Target type: {profile.target_type.value} | Risk: {profile.risk_level} | Confidence: {profile.confidence_score:.0%}")
            # Persist to session state
            try:
                await ctx.set_state(f"ide_profile:{target}", profile.to_dict())
            except Exception:
                logger.debug("Failed to persist target profile to session state", exc_info=True)
        else:
            # Even when restoring from session state, check for newer cached scans
            cached_scans = _collect_cached_scans(ctx.session_id, target)
            if cached_scans:
                profile = _enrich_profile_from_cache(profile, cached_scans)
                await ctx.info(f"📦 Injected {len(cached_scans)} cached scan(s) into restored profile")
            await ctx.info(f"🎯 Target type: {profile.target_type.value} | Risk: {profile.risk_level} | Confidence: {profile.confidence_score:.0%}")

        chain = await loop.run_in_executor(None, lambda: _ide.create_attack_chain(profile, objective))
        await ctx.report_progress(100, 100)
        await ctx.info(f"✅ Attack chain ready: {len(chain.steps)} steps | Risk: {chain.risk_level}")
        return chain.to_dict()

    # ========================================================================
    # validate_environment — check which external binaries are available
    # ========================================================================

    _TOOL_BINARY_MAP = {
        "airmon_ng": "airmon-ng", "airodump_ng": "airodump-ng",
        "aireplay_ng": "aireplay-ng", "aircrack_ng": "aircrack-ng",
        "hcxdumptool": "hcxdumptool", "wifite": "wifite", "wifite2": "wifite2",
        "hcxpcapngtool": "hcxpcapngtool", "eaphammer": "eaphammer",
        "bettercap_wifi": "bettercap", "mdk4": "mdk4",
        "amass": "amass", "subfinder": "subfinder", "autorecon": "autorecon",
        "theharvester": "theHarvester", "dnsenum": "dnsenum", "fierce": "fierce",
        "whois": "whois",
        "nmap": "nmap", "nmap_advanced": "nmap", "masscan": "masscan",
        "rustscan": "rustscan", "arp_scan": "arp-scan",
        "nikto": "nikto", "sqlmap": "sqlmap", "wpscan": "wpscan",
        "dalfox": "dalfox", "jaeles": "jaeles", "xsser": "xsser", "zap": "zaproxy",
        "gobuster": "gobuster", "ffuf": "ffuf", "feroxbuster": "feroxbuster",
        "dirsearch": "dirsearch", "dirb": "dirb", "wfuzz": "wfuzz",
        "dotdotpwn": "dotdotpwn",
        "hydra": "hydra", "hashcat": "hashcat", "john": "john",
        "medusa": "medusa", "patator": "patator", "hashid": "hashid",
        "ophcrack": "ophcrack",
        "enum4linux": "enum4linux", "enum4linux_ng": "enum4linux-ng",
        "netexec": "netexec", "rpcclient": "rpcclient", "smbmap": "smbmap",
        "nbtscan": "nbtscan",
        "metasploit": "msfconsole", "msfvenom": "msfvenom",
        "searchsploit": "searchsploit", "exploit_db": "searchsploit",
        "pwntools": "python3",
        "katana": "katana", "hakrawler": "hakrawler", "gau": "gau",
        "waybackurls": "waybackurls", "httpx": "httpx", "wafw00f": "wafw00f",
        "arjun": "arjun", "paramspider": "paramspider", "x8": "x8",
        "prowler": "prowler", "trivy": "trivy", "kube_hunter": "kube-hunter",
        "kube_bench": "kube-bench", "checkov": "checkov", "terrascan": "terrascan",
        "ropgadget": "ROPgadget", "ropper": "ropper", "one_gadget": "one_gadget",
        "volatility": "volatility", "volatility3": "vol3", "gdb": "gdb",
        "radare2": "r2", "strings": "strings", "objdump": "objdump",
        "checksec": "checksec", "binwalk": "binwalk", "ghidra": "ghidra",
        "angr": "angr", "xxd": "xxd", "mysql": "mysql", "sqlite": "sqlite3",
        "exiftool": "exiftool", "foremost": "foremost", "steghide": "steghide",
        "hashpump": "hashpump", "anew": "anew", "uro": "uro",
        "nuclei": "nuclei", "responder": "responder",
        "jwt_analyzer": "jwt_analyzer", "autopsy": "autopsy",
        "raw_tcp": "", "execute_code": "",
        "browser_fetch": "", "browser_screenshot": "", "browser_eval": "",
        "sherlock": "sherlock", "spiderfoot": "spiderfoot",
        "sublist3r": "sublist3r", "parsero": "parsero",
        "testssl": "testssl", "whatweb": "whatweb", "commix": "commix",
        "joomscan": "joomscan",
        "vulnx": "vulnx",
        "impacket": "impacket", "ldapdomaindump": "ldapdomaindump",
        "adidnsdump": "adidnsdump", "certipy": "certipy",
        "certipy_ad": "certipy", "mitm6": "mitm6",
        "pywerview": "pywerview", "bloodhound": "bloodhound",
        "bloodhound_python": "bloodhound",
    }

    @mcp.tool(
        description="Validate which external binaries are available on this system. Returns a report of installed, missing, and deprecated tools.",
        annotations={"readOnlyHint": True, "openWorldHint": False},
        task=True,
        timeout=None,
    )
    async def validate_environment(
        ctx: Context,
        tool_filter: str = "",
    ) -> Dict[str, Any]:
        """
        Check which external binaries are installed and functional.

        Args:
            tool_filter: Optional comma-separated list of tool names to check.
                         If empty, all registered tools are validated.

        Returns:
            Environment validation report with per-tool status, binary paths, versions
        """
        await ctx.info("🔍 Validating tool environment...")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()

        # Determine which tools to check
        if tool_filter:
            requested = {t.strip().lower() for t in tool_filter.split(",") if t.strip()}
            tools_to_check = [t for t in DIRECT_TOOLS if t in requested]
            if not tools_to_check:
                unknown = requested - set(DIRECT_TOOLS.keys())
                return {"success": False, "error": f"No registered tools match filter", "unknown_tools": sorted(unknown)}
        else:
            tools_to_check = list(DIRECT_TOOLS.keys())

        await ctx.info(f"📋 Validating {len(tools_to_check)} tools...")

        def _check_binary(binary: str) -> Dict[str, Any]:
            import re, shutil, subprocess
            _VERSION_OVERRIDES = {
                "dalfox": "version",
                "httpx": "-version",
            }
            path = shutil.which(binary)
            if not path:
                return {"present": False, "path": None, "version": None}
            version = None
            flags = (_VERSION_OVERRIDES.get(binary, "--version"), "-v", "version")
            for flag in flags:
                try:
                    r = subprocess.run([path, flag], capture_output=True, text=True, timeout=3)
                    output = ((r.stderr or "") + (r.stdout or ""))[:800]
                    best_line = None
                    for ln in output.split("\n"):
                        stripped = ln.strip()
                        if not stripped or len(stripped) <= 3 or path in stripped:
                            continue
                        m = re.search(r'v?\d+\.\d+\.\d+', stripped)
                        if m:
                            version = m.group(0)
                            break
                        if best_line is None:
                            best_line = stripped[:150]
                    if version:
                        break
                    if best_line:
                        version = best_line
                        break
                except (subprocess.TimeoutExpired, OSError, subprocess.SubprocessError):
                    continue
            return {"present": True, "path": path, "version": version}

        results = {}
        sorted_tools = sorted(tools_to_check)
        # Check binaries concurrently in batches of 10
        batch_size = 10
        for batch_start in range(0, len(sorted_tools), batch_size):
            batch = sorted_tools[batch_start:batch_start + batch_size]
            tasks = []
            for tool_name in batch:
                binary = _TOOL_BINARY_MAP.get(tool_name, tool_name)
                tasks.append(loop.run_in_executor(None, _check_binary, binary))
            batch_results = await asyncio.gather(*tasks)
            for tool_name, info in zip(batch, batch_results):
                binary = _TOOL_BINARY_MAP.get(tool_name, tool_name)
                results[tool_name] = {
                    "binary": binary,
                    "present": info["present"],
                    "path": info["path"],
                    "version": info["version"],
                }
            await ctx.report_progress(int(min(batch_start + batch_size, len(sorted_tools)) / len(sorted_tools) * 100), 100)

        await ctx.report_progress(100, 100)

        present = sum(1 for v in results.values() if v["present"])
        missing = sum(1 for v in results.values() if not v["present"])

        await ctx.info(f"✅ Validation complete: {present} present, {missing} missing")

        return {
            "success": True,
            "total": len(results),
            "present": present,
            "missing": missing,
            "tools": results,
        }

    typed_tools_registered = 0
    for public_name in sorted(DIRECT_TOOLS):
        if public_name in _PRIMITIVE_TOOLS:
            continue
        tool_def = _get_registry_tool_definition(public_name)
        if not tool_def:
            continue
        wrapper = _create_typed_tool_wrapper(public_name, tool_def, run_security_tool)
        rich_desc = _build_typed_tool_doc(public_name, tool_def["desc"], tool_def)
        mcp.tool(
            name=public_name,
            description=rich_desc,
            annotations={"readOnlyHint": False, "openWorldHint": True},
        )(wrapper)
        typed_tools_registered += 1

    logger.info(
        f"🚀 {len(DIRECT_TOOLS)} direct routes, {typed_tools_registered} typed MCP tools, + skill bundle helper"
    )

    # ---- Primitive tools (skipped from typed registration, registered manually) ----
    from mcp_core.misc_direct import _http_request

    @mcp.tool(
        name="http_request",
        description="Generic HTTP client — GET/POST form login, cookie capture, API probing. Wraps curl for request execution.",
        annotations={"readOnlyHint": False, "openWorldHint": True},
    )
    async def http_request_tool(
        url: str,
        method: str = "GET",
        data: str = "",
        cookie: str = "",
        headers: str = "",
        follow_redirects: bool = True,
        max_body_size: int = 5000,
        additional_args: str = "",
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Send an HTTP request and return structured response.
        
        Returns dict with: success, status_code, ok (2xx boolean),
        headers (dict), cookies (dict), body (str), body_truncated (bool),
        body_hex (str, if binary).
        """
        return _http_request({
            "url": url,
            "method": method,
            "data": data,
            "cookie": cookie,
            "headers": headers,
            "follow_redirects": follow_redirects,
            "max_body_size": max_body_size,
            "additional_args": additional_args,
            "timeout": timeout,
        })

    # ========================================================================
    # Resources MCP — health + scan results + CLI feedback
    # ========================================================================

    @mcp.resource("health://server")
    async def server_health_resource() -> str:
        """Server health and runtime statistics."""
        uptime = int(time.time() - _server_start_time)
        return json.dumps({
            "status":         "healthy",
            "server":         "hexstrike-ai-pulse",
            "fastmcp":        "3.2.4",
            "uptime_seconds": uptime,
            "tools_count":    len(DIRECT_TOOLS),
            "type_count":     len(DIRECT_TOOLS),
            "cached_scans":   len(_scan_cache),
            "cache_stats":    _scan_cache.stats(),
            "op_metrics":     _op_metrics.summary(),
        }, indent=2)

    @mcp.resource("scan://{target}/latest")
    async def scan_latest(target: str) -> str:
        """Most recent scan result for a given target across all tools (current session)."""
        ctx = get_context()
        session_id = ctx.session_id
        matches = [
            v for k, v in _scan_cache.items()
            if v.get("target") == target and (k.startswith(f"{session_id}:") or k.startswith("seed:"))
        ]
        if not matches:
            return json.dumps({
                "target":  target,
                "status":  "no_results",
                "message": f"No scan results cached for {target} in current session",
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
        """Cached result for a specific tool + target combination (current session)."""
        ctx = get_context()
        session_id = ctx.session_id
        entry = next(
            (v for k, v in sorted(_scan_cache.items(), reverse=True)
             if v.get("tool") == tool_name
             and v.get("target") == target
             and (k.startswith(f"{session_id}:") or k.startswith("seed:"))),
            None,
        )
        if not entry:
            return json.dumps({
                "target":  target,
                "tool":    tool_name,
                "status":  "no_results",
                "message": f"No cached result for {tool_name} on {target} in current session",
            }, indent=2)

        return json.dumps({
            "target":    target,
            "tool":      tool_name,
            "timestamp": entry["timestamp"],
            "result":    entry["result"],
        }, indent=2)

    @mcp.resource("scan://cache/list")
    async def scan_cache_list() -> str:
        """List cached scan results for the current session."""
        ctx = get_context()
        session_id = ctx.session_id
        entries = [
            {
                "key":       k,
                "tool":      v["tool"],
                "target":    v["target"],
                "timestamp": v["timestamp"],
                "success":   v["result"].get("success", False),
            }
            for k, v in _scan_cache.items()
            if k.startswith(f"{session_id}:") or k.startswith("seed:")
        ]
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return json.dumps({"count": len(entries), "scans": entries}, indent=2)

    @mcp.resource("metrics://tools")
    async def tool_metrics() -> str:
        """Operational metrics: success rates, errors, timeouts, cache, confirmations by tool."""
        return json.dumps(_op_metrics.summary(), indent=2)

    @mcp.resource("telemetry://summary")
    async def telemetry_summary() -> str:
        """Unified telemetry summary: runs, errors, cache, confirmations, prompt suggestions."""
        return json.dumps(_pipeline.summary(), indent=2)

    @mcp.resource("telemetry://recent")
    async def telemetry_recent() -> str:
        """Last 100 telemetry events (tool calls + resource reads)."""
        return json.dumps({"events": _pipeline.recent_events(100)}, indent=2)

    @mcp.resource("telemetry://tools/{tool}")
    async def telemetry_per_tool(tool: str) -> str:
        """Per-tool telemetry stats: runs, success rate, avg duration, errors."""
        return json.dumps(_pipeline.per_tool(tool), indent=2)

    @mcp.resource("errors://statistics")
    async def error_statistics() -> str:
        """Error classification statistics from the IntelligentErrorHandler."""
        from server_core.singletons import error_handler as _eh
        return json.dumps(_eh.get_error_statistics(), indent=2)

    @mcp.resource("targets://")
    async def list_targets() -> str:
        """List all known targets with summary findings counts."""
        ts = get_target_store()
        return json.dumps(ts.get_all_targets(), indent=2)

    @mcp.resource("target://{target}")
    async def get_target(target: str) -> str:
        """Full target profile including findings, sessions, and tool history."""
        ts = get_target_store()
        profile = ts.get_target(target)
        if profile is None:
            return json.dumps({"error": f"Unknown target: {target}"}, indent=2)
        return json.dumps(profile, indent=2)

    @mcp.resource("target://{target}/findings")
    async def get_target_findings(target: str) -> str:
        """Findings only for a target (ports, services, technologies, vulnerabilities)."""
        ts = get_target_store()
        profile = ts.get_target(target)
        if profile is None:
            return json.dumps({"error": f"Unknown target: {target}"}, indent=2)
        return json.dumps(profile.get("findings", {}), indent=2)

    @mcp.resource("target://{target}/sessions")
    async def get_target_sessions(target: str) -> str:
        """Session history for a target."""
        ts = get_target_store()
        profile = ts.get_target(target)
        if profile is None:
            return json.dumps({"error": f"Unknown target: {target}"}, indent=2)
        return json.dumps(profile.get("sessions", []), indent=2)

    logger.info("📦 Resources: health://server, scan://{target}/{tool}, metrics://tools, telemetry://summary, telemetry://recent, telemetry://tools/{tool}, errors://statistics, targets://, target://{target}")

    # ========================================================================
    # server_health MCP tool — wraps the health resource for tool-based access
    # ========================================================================
    @mcp.tool(
        description="HexStrike server health, runtime stats, op metrics — call FIRST to check if the MCP server is alive and get a summary of system state, tool count, cache stats, and operational metrics. No parameters needed.",
        annotations={"readOnlyHint": True, "openWorldHint": False},
        task=True,
        timeout=None,
    )
    async def server_health() -> Dict[str, Any]:
        """Server health, runtime statistics, operational metrics. Call first to verify connectivity and get system status summary."""
        uptime = int(time.time() - _server_start_time)
        return {
            "status":         "healthy",
            "server":         "hexstrike-ai-pulse",
            "fastmcp":        "3.2.4",
            "uptime_seconds": uptime,
            "tools_count":    len(DIRECT_TOOLS),
            "cached_scans":   len(_scan_cache),
            "cache_stats":    _scan_cache.stats(),
            "op_metrics":     _op_metrics.summary(),
        }

    # ========================================================================
    # Tool lifecycle MCP tools — install/uninstall/status
    # ========================================================================

    @mcp.tool(
        description="Check if a HexStrike tool's binary is installed on this system. Returns status (installed/not_found/unknown), binary name, category, and install hints. Call with no arguments to list ALL tools and their status.",
        annotations={"readOnlyHint": True, "openWorldHint": False},
        task=True,
        timeout=None,
    )
    async def tool_status(
        ctx: Context,
        tool_name: str = "",
    ) -> Dict[str, Any]:
        """Check installation status of HexStrike tools.

        Args:
            tool_name: Optional tool name (e.g. 'nmap'). If empty, returns
                       summary of all tools (counts by status).

        Returns:
            Tool status with binary path info and install hints.
        """
        if tool_name:
            return _registry.get_tool_status(tool_name)

        available = _registry.get_available()
        missing = _registry.get_missing()
        return {
            "total_tools": len(_registry.all_tool_names),
            "available": len(available),
            "missing": len(missing),
            "available_tools": [t["name"] for t in available],
            "missing_tools": [t["name"] for t in missing],
        }

    @mcp.tool(
        description="Install a HexStrike tool's system binary. Auto-detects apt/pip/gem package manager. Returns install result with command details. Non-destructive — checks if already installed first.",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        task=True,
        timeout=180000,
    )
    async def install_tool(
        ctx: Context,
        tool_name: str,
    ) -> Dict[str, Any]:
        """Install a HexStrike tool's binary.

        Args:
            tool_name: Tool name (e.g. 'nmap', 'sqlmap', 'gobuster')

        Returns:
            Install result with success/error status and command details.
        """
        await ctx.info(f"📦 Checking status of {tool_name}")
        return _registry.install(tool_name)

    # ========================================================================
    # scan_background — async background scan with task protocol
    # ========================================================================

    @mcp.tool(
        task=TaskConfig(mode="optional", poll_interval=timedelta(seconds=10)),
        timeout=None,
        description=(
            "Full background reconnaissance scan on a target. Runs security tools "
            "based on intensity level and returns surface analysis + vulnerability "
            "findings + attack plan in one response.\n\n"
            "Use scan_background() for scans that may take >30s (intensity=medium/full, "
            "CIDR ranges, multiple tools). Returns a task_id immediately via the "
            "background task protocol — the agent can continue working while the scan "
            "runs in the background. Use scan() for quick targeted scans.\n\n"
            "Intensity levels:\n"
            "- quick (default): nmap + whatweb — open ports and tech detection. ~30s.\n"
            "- medium: + nuclei + nikto — adds vulnerability scanning. ~2-3 min.\n"
            "- full: + gobuster (web targets) — complete recon. ~5-10 min.\n\n"
            "Uses scan cache — recently scanned targets return instantly. "
            "Returns {target, intensity, tools, surface, findings, plan, summary}."
        ),
    )
    async def scan_background(
        ctx: Context,
        target: str = "",
        intensity: str = "quick",
        objective: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Run a scan in the background — returns task_id immediately, agent can poll.

        Unlike scan() which blocks until complete, scan_background() returns a task_id
        right away and runs tools sequentially in a background FastMCP task.
        Uses ctx.report_progress() to show progress between tools.
        Best for intensity=medium/full, CIDR ranges, or any scan >30s.

        Returns: task_id (for polling), target, intensity, tools_to_run, status.
        Poll with get_scan_status() or check get_active_tools() for running tasks.

        Do NOT use for quick scans (<30s) — use scan() which blocks and returns faster.
        Do NOT use when you need results immediately — use scan() with intensity=quick instead.

        Example: scan_background('192.168.1.165', intensity='full')
        Next: check get_active_tools() to monitor, or continue with other work
        """
        # Lazy imports to avoid circular dependency (pulse_app imports from us)
        from pulse_app import (
            get_scope, get_surface, get_findings, get_plan,
            _cache_for_target, _suggest_next_from_context,
            TOOLS_BY_INTENSITY,
            _TOOLS_NEED_URL, _TOOLS_NEED_URL_AS_TARGET, _TOOLS_NEED_HOST,
        )
        from urllib.parse import urlparse

        # Resolve target
        scope_data = get_scope(target) if target else get_scope()
        resolved = scope_data.get("active_target") or target
        if not resolved:
            return {"error": "No target specified or found in scope", "target": None}

        intensity = str(intensity).lower()
        if intensity not in TOOLS_BY_INTENSITY:
            intensity = "quick"

        tools_to_run = TOOLS_BY_INTENSITY[intensity]
        tool_results: Dict[str, Any] = {}
        total = len(tools_to_run)

        await ctx.info(f"🎯 Scan {intensity} starting on {resolved} ({total} tools)")

        for idx, tool_name in enumerate(tools_to_run):
            # Check cache — invalidate stale/partial entries
            cache_entries = _cache_for_target(resolved)
            cached_match = None
            for c in cache_entries:
                if str(c.get("tool", "")).lower() == tool_name:
                    cached_match = c
                    break
            if cached_match:
                stdout = (cached_match.get("result", {}) or {}).get("stdout", "")
                status = cached_match.get("status", "")
                stale = False
                if status in ("timeout", "failed"):
                    stale = True
                elif tool_name in _TOOLS_NEED_HOST and "0 hosts up" in stdout:
                    stale = True
                if stale:
                    await ctx.info(f"📊 [{idx+1}/{total}] {tool_name} — stale cache ({status}), re-running")
                else:
                    tool_results[tool_name] = {"status": "cached", "cached": True}
                    await ctx.info(f"📊 [{idx+1}/{total}] {tool_name} — cached, skipping")
                    continue

            params: Dict[str, Any] = {"target": resolved}
            if tool_name in _TOOLS_NEED_URL:
                if not resolved.startswith(("http://", "https://")):
                    params = {"url": f"http://{resolved}"}
                else:
                    params = {"url": resolved}
            elif tool_name in _TOOLS_NEED_URL_AS_TARGET:
                if not resolved.startswith(("http://", "https://")):
                    params = {"url": f"http://{resolved}", "target": resolved}
                else:
                    params = {"url": resolved, "target": resolved}
            elif tool_name in _TOOLS_NEED_HOST:
                host = urlparse(resolved).hostname or resolved
                params = {"target": host, "scan_type": "-sTV"}

            # Apply parameter optimizer for tool-level tuning (--timeout, threads, additional_args)
            # Additive only — caller_keys protected, never overrides scan() built params
            params = _optimizer.optimize(tool_name, params)

            await ctx.info(f"🔨 Running {tool_name} on {resolved}")
            try:
                from mcp_core.null_context import NullContext
                null_ctx = NullContext()
                out = await run_security_tool(null_ctx, tool_name, params)
                ok = out.get("success", False)
                tool_results[tool_name] = {
                    "status": "completed" if ok else "failed",
                    "returncode": out.get("returncode"),
                }
                if not ok:
                    tool_results[tool_name]["error"] = out.get("error", "Unknown error")
            except Exception as e:
                tool_results[tool_name] = {"status": "error", "error": str(e)}

            await ctx.info(f"📊 [{idx+1}/{total}] {tool_name} — done")

        await ctx.info("📊 Post-processing: surface + findings + plan")

        # Post-scan analysis
        surface_data = get_surface(resolved)
        findings_data = get_findings(resolved) if intensity in ("medium", "full") else []
        plan_data = get_plan(resolved, objective) if intensity == "full" else {
            "target": resolved, "steps": [], "step_count": 0,
            "summary": "Skipped — use full intensity for planning",
        }

        # TargetStore for MCP Resources
        try:
            ts = get_target_store()
            ts.record_scan(
                target=resolved,
                tools_used=list(tool_results.keys()),
                surface_data=surface_data,
                findings=findings_data,
            )
        except Exception as e:
            await ctx.warning(f"TargetStore record_scan failed: {e}")

        suggestion = _suggest_next_from_context(surface_data, findings_data)
        result = {
            "target": resolved,
            "intensity": intensity,
            "tools": tool_results,
            "surface": surface_data,
            "findings": findings_data,
            "plan": plan_data,
            "summary": (
                f"Scan {intensity} on {resolved}: "
                f"{surface_data.get('ports_count', 0)} ports, "
                f"{len(findings_data)} findings, "
                f"{plan_data.get('step_count', 0)} plan steps"
            ),
        }
        if suggestion:
            result["next_suggested_tool"] = suggestion
        return result

    # ========================================================================
    # Workflow Prompts — native MCP prompts for multi-tool attack chains
    # ========================================================================

    from mcp_core.prompts import register_prompts
    from mcp_core.ctf_engine import register_ctf_tools
    from mcp_core.cve_engine import register_cve_tools
    from mcp_core.bugbounty_engine import register_bugbounty_tools
    register_prompts(mcp)
    register_ctf_tools(mcp)
    register_cve_tools(mcp)
    register_bugbounty_tools(mcp)
    logger.info("🎯 Workflow prompts registered: bug_bounty_recon, wifi_attack_chain, ctf_web_challenge, smb_lateral_movement, cloud_security_audit")

    # Add pulse_app provider so its tools (scan, get_surface, etc.) are
    # visible through the RegexSearchTransform alongside exec tools.
    try:
        from pulse_app import app as _pulse_app
        mcp.add_provider(_pulse_app)
        logger.info("📊 Pulse dashboard tools registered")
    except Exception as exc:
        logger.warning("⚠️ Could not register pulse_app provider: %s — scan, get_live_dashboard, pulse_dashboard will be missing from always_visible", exc)

    # Background cache warmup — pre-seed version markers for installed tools
    try:
        _start_warmup(logger)
    except Exception:
        pass  # warmup is best-effort

    return mcp


def _start_warmup(logger: Any = None) -> None:
    """Background cache warmup: seed version markers for commonly installed tools."""
    import subprocess as _subprocess

    _log = logger or logging.getLogger(__name__)

    def _warm() -> None:
        try:
            available = _registry.get_available()[:15]  # limit to first 15
            for t in available:
                binary = t.get("binary", t["name"])
                try:
                    proc = _subprocess.run(
                        [binary, "--version"],
                        capture_output=True, text=True, timeout=3,
                        start_new_session=True,
                    )
                    if proc.returncode == 0:
                        _scan_cache.set(
                            f"seed:{t['name']}:version",
                            {
                                "tool": t["name"],
                                "target": "version",
                                "result": {
                                    "success": True,
                                    "output": proc.stdout.strip()[:200],
                                },
                                "timestamp": time.time(),
                            },
                            execution_time=0.5,
                        )
                except (_subprocess.TimeoutExpired, OSError, _subprocess.SubprocessError):
                    pass
            _log.debug(f"🌡️ Cache warmup complete ({len(available)} tools checked)")
        except Exception as e:
            _log.warning(f"🌡️ Cache warmup failed: {e}")

    threading.Thread(target=_warm, daemon=True).start()
