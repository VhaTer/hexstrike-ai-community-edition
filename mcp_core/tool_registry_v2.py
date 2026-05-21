"""Dynamic tool registry with lifecycle management.

Replaces static TOOL_ROUTES dict with a dynamic registry that:
- Tracks binary install status (installed / not_found / unknown)
- Gracefully degrades when tools are missing
- Supports install + uninstall lifecycle via MCP tools
- Provides install hints for common tools
"""

import shutil
import subprocess
import threading
from typing import Any

from mcp_core.tool_routes import TOOL_ROUTES
from tool_registry import TOOLS

# Install hints for common tools: binary -> (method, package)
_INSTALL_HINTS: dict[str, tuple[str, str]] = {
    "nmap": ("apt", "nmap"),
    "masscan": ("apt", "masscan"),
    "arp-scan": ("apt", "arp-scan"),
    "whatweb": ("gem", "whatweb"),
    "nikto": ("apt", "nikto"),
    "sqlmap": ("pip", "sqlmap"),
    "gobuster": ("apt", "gobuster"),
    "ffuf": ("pip", "ffuf"),
    "feroxbuster": ("pip", "feroxbuster"),
    "dirb": ("apt", "dirb"),
    "hydra": ("apt", "hydra"),
    "hashcat": ("apt", "hashcat"),
    "john": ("apt", "john"),
    "medusa": ("apt", "medusa"),
    "netexec": ("pip", "netexec"),
    "enum4linux": ("apt", "enum4linux"),
    "smbmap": ("pip", "smbmap"),
    "metasploit": ("apt", "metasploit-framework"),
    "searchsploit": ("apt", "exploitdb"),
    "wpscan": ("gem", "wpscan"),
    "dalfox": ("go", "dalfox"),
    "nuclei": ("pip", "nuclei"),
    "testssl": ("apt", "testssl.sh"),
    "whois": ("apt", "whois"),
    "dnsenum": ("apt", "dnsenum"),
    "fierce": ("apt", "fierce"),
    "amass": ("apt", "amass"),
    "subfinder": ("apt", "subfinder"),
    "tcpdump": ("apt", "tcpdump"),
    "tshark": ("apt", "tshark"),
    "airmon-ng": ("apt", "aircrack-ng"),
    "airodump-ng": ("apt", "aircrack-ng"),
    "aireplay-ng": ("apt", "aircrack-ng"),
    "aircrack-ng": ("apt", "aircrack-ng"),
    "hcxdumptool": ("apt", "hcxdumptool"),
    "hcxpcapngtool": ("apt", "hcxpcapngtool"),
    "wifite2": ("pip", "wifite2"),
    "bettercap": ("apt", "bettercap"),
    "mdk4": ("apt", "mdk4"),
    "hashid": ("pip", "hashid"),
    "ophcrack": ("apt", "ophcrack"),
    "exploit_db": ("apt", "exploitdb"),
    "volatility": ("pip", "volatility"),
    "volatility3": ("pip", "volatility3"),
    "gdb": ("apt", "gdb"),
    "radare2": ("apt", "radare2"),
    "strings": ("apt", "binutils"),
    "binwalk": ("pip", "binwalk"),
    "xxd": ("apt", "xxd"),
    "exiftool": ("apt", "exiftool"),
    "foremost": ("apt", "foremost"),
    "steghide": ("apt", "steghide"),
    "responder": ("pip", "responder"),
    "bbot": ("pip", "bbot"),
}

# Tool names that are NOT real binaries — these are internal aliases
_SKIP_INSTALL = {"nmap_advanced", "nxc", "bettercap_wifi", "wifite2"}


class ToolRegistry:
    """Dynamic registry of HexStrike tools with install status detection."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._routes: dict[str, tuple[str, str]] = {}
        self._names_by_binary: dict[str, list[str]] = {}
        self._status_cache: dict[str, str] = {}
        self._build_manifests()

    def _build_manifests(self) -> None:
        for name, (mod_path, func_name, binary) in TOOL_ROUTES.items():
            self._routes[name] = (func_name, binary)
            self._names_by_binary.setdefault(binary, []).append(name)

    @property
    def all_tool_names(self) -> list[str]:
        return sorted(self._routes)

    def refresh_status(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for binary in self._names_by_binary:
            found = shutil.which(binary) is not None
            s = "installed" if found else "not_found"
            self._status_cache[binary] = s
            statuses[binary] = s
        return statuses

    def _get_binary_status(self, binary: str) -> str:
        if binary not in self._status_cache:
            self._status_cache[binary] = (
                "installed" if shutil.which(binary) else "not_found"
            )
        return self._status_cache[binary]

    def get_tool_status(self, name: str) -> dict[str, Any]:
        route = self._routes.get(name)
        if not route:
            return {"name": name, "status": "unknown", "error": f"No route for {name}"}
        _, binary = route
        s = self._get_binary_status(binary)
        tool_def = TOOLS.get(name.replace("_", "-"))
        return {
            "name": name,
            "binary": binary,
            "status": s,
            "category": tool_def.get("category", "") if tool_def else "",
            "desc": tool_def.get("desc", "") if tool_def else "",
            "install_hint": self._get_install_hint(binary),
        }

    def get_available(self) -> list[dict[str, Any]]:
        return [s for name in sorted(self._routes)
                if (s := self.get_tool_status(name))["status"] == "installed"]

    def get_missing(self) -> list[dict[str, Any]]:
        return [s for name in sorted(self._routes)
                if (s := self.get_tool_status(name))["status"] == "not_found"]

    def _get_install_hint(self, binary: str) -> str:
        hint = _INSTALL_HINTS.get(binary)
        if hint:
            method, pkg = hint
            if method == "apt":
                return f"apt install {pkg}"
            if method == "pip":
                return f"pip install {pkg}"
            if method == "gem":
                return f"gem install {pkg}"
            if method == "go":
                return f"go install {pkg}"
        return f"Install '{binary}' manually (no auto-install hint)"

    def _get_auto_install_cmd(self, binary: str) -> list[str] | None:
        hint = _INSTALL_HINTS.get(binary)
        if not hint:
            return None
        method, pkg = hint
        if method == "apt":
            return ["sudo", "apt", "install", "-y", pkg]
        if method == "pip":
            return ["pip", "install", pkg]
        if method == "gem":
            return ["gem", "install", pkg]
        return None

    def install(self, name: str) -> dict[str, Any]:
        if name in _SKIP_INSTALL:
            return {"success": False, "error": f"{name} is an alias tool — install the base tool instead"}
        status = self.get_tool_status(name)
        if status["status"] == "installed":
            return {"success": True, "message": f"{name} already installed", "skipped": True}
        if status["status"] == "unknown":
            return {"success": False, "error": f"Unknown tool: {name}"}

        binary = status["binary"]
        cmd = self._get_auto_install_cmd(binary)
        if not cmd:
            hint = self._get_install_hint(binary)
            return {"success": False, "install_command": hint,
                    "error": f"No auto-install method. Run: {hint}"}

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                self._status_cache.pop(binary, None)
                return {"success": True, "message": f"Installed {name}", "command": " ".join(cmd)}
            return {"success": False, "error": f"Install failed: {proc.stderr[:200]}",
                    "command": " ".join(cmd)}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Install timed out (120s)"}
        except FileNotFoundError:
            hint = self._get_install_hint(binary)
            return {"success": False, "install_command": hint,
                    "error": f"Package manager unavailable. Run: {hint}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)[:200]}


_registry = ToolRegistry()
