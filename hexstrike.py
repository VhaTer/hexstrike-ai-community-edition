#!/usr/bin/env python3
"""
hexstrike — CLI for HexStrike AI-PULSE

Usage:
  hexstrike serve     Start the Pulse HTTP/SSE server
  hexstrike scan      Run a tool directly (no server needed)
  hexstrike tools     List available tools
  hexstrike status    Check server health
  hexstrike validate  Validate environment (binary availability)
  hexstrike mcp       Run MCP stdio bridge (Claude Desktop)
  hexstrike ctf       CTF workflow analysis

Flags:
  --version     Show version
  --help        Show this message and exit
"""

import argparse
import importlib
import io
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("hexstrike")

VERSION = "0.8.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888

# ---------------------------------------------------------------------------
# Direct tool routing (same pattern as DIRECT_TOOLS in server_setup.py)
# ---------------------------------------------------------------------------
DIRECT_ROUTES = {
    # wifi
    "airmon_ng":         ("mcp_core.wifi_direct", "wifi_exec", "airmon_ng"),
    "airodump_ng":       ("mcp_core.wifi_direct", "wifi_exec", "airodump_ng"),
    "aireplay_ng":       ("mcp_core.wifi_direct", "wifi_exec", "aireplay_ng"),
    "aircrack_ng":       ("mcp_core.wifi_direct", "wifi_exec", "aircrack_ng"),
    "hcxdumptool":       ("mcp_core.wifi_direct", "wifi_exec", "hcxdumptool"),
    "hcxpcapngtool":     ("mcp_core.wifi_direct", "wifi_exec", "hcxpcapngtool"),
    "eaphammer":         ("mcp_core.wifi_direct", "wifi_exec", "eaphammer"),
    "wifite":            ("mcp_core.wifi_direct", "wifi_exec", "wifite2"),
    "airbase_ng":        ("mcp_core.wifi_direct", "wifi_exec", "airbase_ng"),
    "airdecap_ng":       ("mcp_core.wifi_direct", "wifi_exec", "airdecap_ng"),
    "bettercap":         ("mcp_core.wifi_direct", "wifi_exec", "bettercap_wifi"),
    "tcpdump":           ("mcp_core.wifi_direct", "wifi_exec", "tcpdump"),
    "tshark":            ("mcp_core.wifi_direct", "wifi_exec", "tshark"),
    "mdk4":              ("mcp_core.wifi_direct", "wifi_exec", "mdk4"),
    # recon
    "amass":             ("mcp_core.recon_direct", "recon_exec", "amass"),
    "subfinder":         ("mcp_core.recon_direct", "recon_exec", "subfinder"),
    "autorecon":         ("mcp_core.recon_direct", "recon_exec", "autorecon"),
    "theharvester":      ("mcp_core.recon_direct", "recon_exec", "theharvester"),
    "dnsenum":           ("mcp_core.recon_direct", "recon_exec", "dnsenum"),
    "fierce":            ("mcp_core.recon_direct", "recon_exec", "fierce"),
    "whois":             ("mcp_core.recon_direct", "recon_exec", "whois"),
    # net_scan
    "nmap":              ("mcp_core.net_scan_direct", "net_scan_exec", "nmap"),
    "nmap_advanced":     ("mcp_core.net_scan_direct", "net_scan_exec", "nmap-advanced"),
    "masscan":           ("mcp_core.net_scan_direct", "net_scan_exec", "masscan"),
    "rustscan":          ("mcp_core.net_scan_direct", "net_scan_exec", "rustscan"),
    "arp_scan":          ("mcp_core.net_scan_direct", "net_scan_exec", "arp-scan"),
    # web_scan
    "nikto":             ("mcp_core.web_scan_direct", "web_scan_exec", "nikto"),
    "sqlmap":            ("mcp_core.web_scan_direct", "web_scan_exec", "sqlmap"),
    "wpscan":            ("mcp_core.web_scan_direct", "web_scan_exec", "wpscan"),
    "dalfox":            ("mcp_core.web_scan_direct", "web_scan_exec", "dalfox"),
    "jaeles":            ("mcp_core.web_scan_direct", "web_scan_exec", "jaeles"),
    "xsser":             ("mcp_core.web_scan_direct", "web_scan_exec", "xsser"),
    "zap":               ("mcp_core.web_scan_direct", "web_scan_exec", "zap"),
    # web_fuzz
    "gobuster":          ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "gobuster"),
    "ffuf":              ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "ffuf"),
    "feroxbuster":       ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "feroxbuster"),
    "dirsearch":         ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "dirsearch"),
    "dirb":              ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "dirb"),
    "wfuzz":             ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "wfuzz"),
    "dotdotpwn":         ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "dotdotpwn"),
    # password_cracking
    "hydra":             ("mcp_core.password_cracking_direct", "pwdcrack_exec", "hydra"),
    "hashcat":           ("mcp_core.password_cracking_direct", "pwdcrack_exec", "hashcat"),
    "john":              ("mcp_core.password_cracking_direct", "pwdcrack_exec", "john"),
    "medusa":            ("mcp_core.password_cracking_direct", "pwdcrack_exec", "medusa"),
    "patator":           ("mcp_core.password_cracking_direct", "pwdcrack_exec", "patator"),
    "hashid":            ("mcp_core.password_cracking_direct", "pwdcrack_exec", "hashid"),
    "ophcrack":          ("mcp_core.password_cracking_direct", "pwdcrack_exec", "ophcrack"),
    # smb_enum
    "enum4linux":        ("mcp_core.smb_enum_direct", "smb_enum_exec", "enum4linux"),
    "enum4linux-ng":     ("mcp_core.smb_enum_direct", "smb_enum_exec", "enum4linux-ng"),
    "netexec":           ("mcp_core.smb_enum_direct", "smb_enum_exec", "netexec"),
    "rpcclient":         ("mcp_core.smb_enum_direct", "smb_enum_exec", "rpcclient"),
    "smbmap":            ("mcp_core.smb_enum_direct", "smb_enum_exec", "smbmap"),
    "nbtscan":           ("mcp_core.smb_enum_direct", "smb_enum_exec", "nbtscan"),
    "nxc":               ("mcp_core.smb_enum_direct", "smb_enum_exec", "nxc"),
    "evil_winrm":        ("mcp_core.smb_enum_direct", "smb_enum_exec", "evil_winrm"),
    # exploit
    "metasploit":        ("mcp_core.exploit_framework_direct", "exploit_exec", "metasploit"),
    "msfvenom":          ("mcp_core.exploit_framework_direct", "exploit_exec", "msfvenom"),
    "searchsploit":      ("mcp_core.exploit_framework_direct", "exploit_exec", "exploit_db"),
    "exploit_db":        ("mcp_core.exploit_framework_direct", "exploit_exec", "exploit_db"),
    "pwntools":          ("mcp_core.exploit_framework_direct", "exploit_exec", "pwntools"),
    # web_recon
    "katana":            ("mcp_core.web_recon_direct", "web_recon_exec", "katana"),
    "hakrawler":         ("mcp_core.web_recon_direct", "web_recon_exec", "hakrawler"),
    "gau":               ("mcp_core.web_recon_direct", "web_recon_exec", "gau"),
    "waybackurls":       ("mcp_core.web_recon_direct", "web_recon_exec", "waybackurls"),
    "httpx":             ("mcp_core.web_recon_direct", "web_recon_exec", "httpx"),
    "wafw00f":           ("mcp_core.web_recon_direct", "web_recon_exec", "wafw00f"),
    "arjun":             ("mcp_core.web_recon_direct", "web_recon_exec", "arjun"),
    "paramspider":       ("mcp_core.web_recon_direct", "web_recon_exec", "paramspider"),
    "x8":                ("mcp_core.web_recon_direct", "web_recon_exec", "x8"),
    # security
    "prowler":           ("mcp_core.security_direct", "security_exec", "prowler"),
    "trivy":             ("mcp_core.security_direct", "security_exec", "trivy"),
    "kube_hunter":       ("mcp_core.security_direct", "security_exec", "kube-hunter"),
    "kube_bench":        ("mcp_core.security_direct", "security_exec", "kube-bench"),
    "checkov":           ("mcp_core.security_direct", "security_exec", "checkov"),
    "terrascan":         ("mcp_core.security_direct", "security_exec", "terrascan"),
    # misc
    "ropgadget":         ("mcp_core.misc_direct", "misc_exec", "ropgadget"),
    "ropper":            ("mcp_core.misc_direct", "misc_exec", "ropper"),
    "one_gadget":        ("mcp_core.misc_direct", "misc_exec", "one_gadget"),
    "volatility":        ("mcp_core.misc_direct", "misc_exec", "volatility"),
    "volatility3":       ("mcp_core.misc_direct", "misc_exec", "volatility3"),
    "gdb":               ("mcp_core.misc_direct", "misc_exec", "gdb"),
    "radare2":           ("mcp_core.misc_direct", "misc_exec", "radare2"),
    "strings":           ("mcp_core.misc_direct", "misc_exec", "strings"),
    "objdump":           ("mcp_core.misc_direct", "misc_exec", "objdump"),
    "checksec":          ("mcp_core.misc_direct", "misc_exec", "checksec"),
    "binwalk":           ("mcp_core.misc_direct", "misc_exec", "binwalk"),
    "ghidra":            ("mcp_core.misc_direct", "misc_exec", "ghidra"),
    "angr":              ("mcp_core.misc_direct", "misc_exec", "angr"),
    "xxd":               ("mcp_core.misc_direct", "misc_exec", "xxd"),
    "mysql":             ("mcp_core.misc_direct", "misc_exec", "mysql"),
    "sqlite":            ("mcp_core.misc_direct", "misc_exec", "sqlite"),
    "exiftool":          ("mcp_core.misc_direct", "misc_exec", "exiftool"),
    "foremost":          ("mcp_core.misc_direct", "misc_exec", "foremost"),
    "steghide":          ("mcp_core.misc_direct", "misc_exec", "steghide"),
    "hashpump":          ("mcp_core.misc_direct", "misc_exec", "hashpump"),
    "anew":              ("mcp_core.misc_direct", "misc_exec", "anew"),
    "uro":               ("mcp_core.misc_direct", "misc_exec", "uro"),
    "nuclei":            ("mcp_core.misc_direct", "misc_exec", "nuclei"),
    "responder":         ("mcp_core.misc_direct", "misc_exec", "responder"),
    "jwt_analyzer":      ("mcp_core.misc_direct", "misc_exec", "jwt_analyzer"),
    "autopsy":           ("mcp_core.misc_direct", "misc_exec", "autopsy"),
    "libc":              ("mcp_core.misc_direct", "misc_exec", "libc"),
    "api_schema_analyzer": ("mcp_core.misc_direct", "misc_exec", "api_schema_analyzer"),
    "graphql_scanner":   ("mcp_core.misc_direct", "misc_exec", "graphql_scanner"),
    "api_fuzzer":        ("mcp_core.misc_direct", "misc_exec", "api_fuzzer"),
    "bbot":              ("mcp_core.misc_direct", "misc_exec", "bbot"),
    "bulk_extractor":    ("mcp_core.misc_direct", "misc_exec", "bulk_extractor"),
    "scalpel":           ("mcp_core.misc_direct", "misc_exec", "scalpel"),
    "falco":             ("mcp_core.misc_direct", "misc_exec", "falco"),
    "qsreplace":         ("mcp_core.misc_direct", "misc_exec", "qsreplace"),
    # web_probe
    "whatweb":           ("mcp_core.web_probe_direct", "web_probe_exec", "whatweb"),
    "commix":            ("mcp_core.web_probe_direct", "web_probe_exec", "commix"),
    "joomscan":          ("mcp_core.web_probe_direct", "web_probe_exec", "joomscan"),
    # testssl
    "testssl":           ("mcp_core.testssl_direct", "testssl_exec", "testssl"),
    # vuln_intel
    "vulnx":             ("mcp_core.vuln_intel_direct", "vuln_intel_exec", "vulnx"),
    # osint
    "sherlock":          ("mcp_core.osint_direct", "osint_exec", "sherlock"),
    "spiderfoot":        ("mcp_core.osint_direct", "osint_exec", "spiderfoot"),
    "sublist3r":         ("mcp_core.osint_direct", "osint_exec", "sublist3r"),
    "parsero":           ("mcp_core.osint_direct", "osint_exec", "parsero"),
    # active_directory
    "impacket":          ("mcp_core.active_directory_direct", "ad_exec", "impacket"),
    "ldapdomaindump":    ("mcp_core.active_directory_direct", "ad_exec", "ldapdomaindump"),
    "adidnsdump":        ("mcp_core.active_directory_direct", "ad_exec", "adidnsdump"),
    "certipy":           ("mcp_core.active_directory_direct", "ad_exec", "certipy_ad"),
    "certipy_ad":        ("mcp_core.active_directory_direct", "ad_exec", "certipy_ad"),
    "mitm6":             ("mcp_core.active_directory_direct", "ad_exec", "mitm6"),
    "pywerview":         ("mcp_core.active_directory_direct", "ad_exec", "pywerview"),
    "bloodhound":        ("mcp_core.active_directory_direct", "ad_exec", "bloodhound"),
    "bloodhound_python": ("mcp_core.active_directory_direct", "ad_exec", "bloodhound"),
}


def _resolve_tool(tool_name: str):
    """Lazy-import and resolve a tool exec function."""
    route = DIRECT_ROUTES.get(tool_name)
    if not route:
        return None
    module_path, exec_name, tool_key = route
    mod = importlib.import_module(module_path)
    exec_func = getattr(mod, exec_name)
    return exec_func, tool_key


def _format_scan_result(tool_name: str, target: str, result: dict) -> str:
    """Format scan result with a compact summary box + raw output."""
    C = _cli_colors()
    b, g, w, R = C['ACCENT_LINE'], C['TERMINAL_GRAY'], C['BRIGHT_WHITE'], C['RESET']
    out = result.get("output", "") or result.get("stdout", "")
    err = result.get("error", "") or result.get("stderr", "")
    success = result.get("success", False)
    timed_out = result.get("timed_out", False)
    exec_time = result.get("execution_time", 0)
    if success:
        icon, sl = "✅", f"Success ({exec_time:.1f}s)"
    elif timed_out:
        icon, sl = "⏰", f"Timed out ({exec_time:.1f}s)"
    else:
        icon, sl = "❌", f"Failed"
    rows = [
        f"  {w}Tool:{R}    {tool_name}",
        f"  {w}Target:{R}  {target or '(none)'}",
        f"  {w}Status:{R}  {icon} {sl}",
    ]
    inner = max(_dw(r) for r in rows)
    def p(s): return f"  {b}│{R}  {s}{' ' * (inner - _dw(s))}  {b}│{R}"
    buf = [f"  {b}╭{'─' * (inner + 4)}╮{R}"]
    for r in rows:
        buf.append(p(r))
    buf.append(f"  {b}╰{'─' * (inner + 4)}╯{R}")
    body = out.strip() if out else ""
    if err:
        body += f"\n{g}Error: {err.strip()[:500]}{R}" if body else f"{g}Error: {err.strip()[:500]}{R}"
    if body:
        buf.append("")
        buf.append(body)
    return '\n'.join(buf) + '\n'


# ============================================================================
# Subcommand: serve
# ============================================================================

def cmd_serve(args):
    """Start the Pulse HTTP/SSE server."""
    from mcp_core.server_setup import setup_mcp_server_standalone
    from server_core.modern_visual_engine import ModernVisualEngine
    server_setup = importlib.import_module("hexstrike_server")

    host = args.host or os.environ.get("HEXSTRIKE_HOST", DEFAULT_HOST)
    port = args.port or int(os.environ.get("HEXSTRIKE_PORT", DEFAULT_PORT))

    print(ModernVisualEngine.create_banner())
    logger.info(f"🚀 Starting HexStrike Pulse Server")
    logger.info(f"📡 Server: http://{host}:{port}")

    mcp = setup_mcp_server_standalone(logger)
    server_setup.register_http_routes(mcp, logger)

    mcp.run(
        transport="http",
        host=host,
        port=port,
        show_banner=False,
    )


# ============================================================================
# Subcommand: scan
# ============================================================================

def cmd_scan(args):
    """Run a security tool directly (no server required)."""
    tool_name = args.tool
    target = args.target

    resolved = _resolve_tool(tool_name)
    if not resolved:
        logger.error(f"❌ Unknown tool: {tool_name}")
        similar = [t for t in DIRECT_ROUTES if tool_name in t or t.startswith(tool_name[:3])]
        if similar:
            logger.info(f"   Did you mean: {', '.join(similar[:5])}?")
        sys.exit(1)

    exec_func, tool_key = resolved

    params = {}
    if target:
        params["target"] = target
        params["url"] = target
    if args.param:
        for p in args.param:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v

    result = exec_func(tool_key, params)

    if args.json:
        output = json.dumps(result, indent=2, default=str)
    else:
        output = _format_scan_result(tool_name, target, result)

    if args.output:
        Path(args.output).write_text(output)

    print(output, end="")


# ============================================================================
# Subcommand: tools
# ============================================================================

def cmd_tools(args):
    """List available tools with their descriptions."""
    from tool_registry import TOOLS

    if args.filter:
        matching = {k: v for k, v in TOOLS.items() if args.filter.lower() in k.lower()}
        if not matching:
            msg = f"No tools matching '{args.filter}'"
            print(msg)
            if args.output:
                Path(args.output).write_text(msg + "\n")
            return
        source = matching
    else:
        source = TOOLS

    cat_order = ["essential", "recon", "web_vuln", "web_fuzz", "password",
                 "smb", "exploit", "osint", "cloud", "binary", "wifi", "misc"]

    def _cat_key(item):
        cat = item[1].get("category", "zzz")
        return (cat_order.index(cat) if cat in cat_order else 99, item[0])

    by_cat: dict[str, list] = {}
    for name, defn in sorted(source.items()):
        cat = defn.get("category", "uncategorized")
        by_cat.setdefault(cat, []).append((name, defn))

    if args.json:
        data = {}
        for cat in cat_order:
            if cat not in by_cat:
                continue
            for name, defn in sorted(by_cat[cat]):
                data[name] = {
                    "category": cat,
                    "description": defn.get("desc", ""),
                    "effectiveness": defn.get("effectiveness", None),
                }
        for cat in sorted(set(by_cat) - set(cat_order)):
            for name, defn in sorted(by_cat[cat]):
                data[name] = {
                    "category": cat,
                    "description": defn.get("desc", ""),
                    "effectiveness": defn.get("effectiveness", None),
                }
        output = json.dumps(data, indent=2)
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            idx = 0
            for cat in cat_order:
                if cat not in by_cat:
                    continue
                print(f"\n── {cat} ──")
                for name, defn in sorted(by_cat[cat]):
                    idx += 1
                    desc = defn.get("desc", "")
                    eff = defn.get("effectiveness", "")
                    eff_str = f" [{eff:.0%}]" if isinstance(eff, (int, float)) else ""
                    print(f"({idx:2d}) {name:22s} {desc}{eff_str}")

            for cat in sorted(set(by_cat) - set(cat_order)):
                print(f"\n── {cat} ──")
                for name, defn in sorted(by_cat[cat]):
                    idx += 1
                    print(f"({idx:2d}) {name:22s} {defn.get('desc', '')}")
        output = buf.getvalue()

    print(output, end="")
    if args.output:
        Path(args.output).write_text(output)


# ============================================================================
# Subcommand: status
# ============================================================================

def _cli_colors():
    from server_core.modern_visual_engine import ModernVisualEngine as _MVE
    return _MVE.COLORS


_ansi_strip = __import__('re').compile(r'\x1b\[[0-9;]*m').sub
_wcwidth = None
def _dw(s: str) -> int:
    global _wcwidth
    if _wcwidth is None:
        from wcwidth import wcswidth as _wcwidth
    return _wcwidth(_ansi_strip('', s))


def cmd_status(args):
    """Check Pulse server health."""
    host = args.host or os.environ.get("HEXSTRIKE_HOST", DEFAULT_HOST)
    port = args.port or int(os.environ.get("HEXSTRIKE_PORT", DEFAULT_PORT))
    url = f"http://{host}:{port}/health"

    try:
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        status = data.get("status", "unknown")
        uptime = data.get("uptime_seconds", 0)
        checks = data.get("checks", {})

        if args.json:
            output = json.dumps({
                "status": status, "uptime_seconds": uptime,
                "host": host, "port": port, "checks": checks,
            }, indent=2)
        else:
            C = _cli_colors()
            b, g, w, R = C['ACCENT_LINE'], C['TERMINAL_GRAY'], C['BRIGHT_WHITE'], C['RESET']
            icon = "🟢" if status == "ready" else "🟡" if status == "degraded" else "🔴"
            tools = checks.get("essential_tools", {})
            disk = checks.get("disk", {})
            rows = [
                f"  {w}Status:{R}   {icon} {status}",
                f"  {w}Uptime:{R}   {uptime // 3600}h{uptime % 3600 // 60}m",
                f"  {w}Tools:{R}    {tools.get('available', '?')}/{tools.get('total', '?')} available",
            ]
            if disk:
                rows.append(f"  {w}Disk:{R}     {disk.get('free_gb', '?')} GB free ({disk.get('usage_pct', '?')}% used)")
            rows.append(f"  {g}{url}{R}")
            inner = max(_dw(r) for r in rows)
            def p(s): return f"  {b}│{R}  {s}{' ' * (inner - _dw(s))}  {b}│{R}"
            buf = [f"  {b}╭{'─' * (inner + 4)}╮{R}"]
            buf.append(p(rows[0]))
            buf.extend(p(r) for r in rows[1:-1])
            buf.append(f"  {b}│{R}  {g}{url}{R}{' ' * (inner - len(url))}  {b}│{R}")
            buf.append(f"  {b}╰{'─' * (inner + 4)}╯{R}")
            output = '\n'.join(buf) + '\n'

        print(output, end="")
        if args.output:
            Path(args.output).write_text(output)
    except urllib.error.URLError:
        msg = f"🔴 Pulse server not responding at {url}\n"
        print(msg, end="")
        if args.output:
            Path(args.output).write_text(msg)
        sys.exit(1)
    except Exception as e:
        msg = f"🔴 Error: {e}\n"
        print(msg, end="")
        if args.output:
            Path(args.output).write_text(msg)
        sys.exit(1)


# ============================================================================
# Subcommand: validate
# ============================================================================

def cmd_validate(args):
    """Validate which external tools are available."""
    import shutil

    tools_to_check = list(DIRECT_ROUTES.keys())
    if args.tool_filter:
        tools_to_check = [t for t in tools_to_check if args.tool_filter.lower() in t.lower()]

    present = []
    missing = []
    for tool_name in sorted(tools_to_check):
        _, _, binary_key = DIRECT_ROUTES[tool_name]
        path = shutil.which(binary_key)
        if path:
            present.append(tool_name)
        else:
            missing.append(tool_name)

    if args.json:
        output = json.dumps({
            "total": len(tools_to_check),
            "present_count": len(present),
            "missing_count": len(missing),
            "present": [{"tool": t, "binary": DIRECT_ROUTES[t][2], "path": shutil.which(DIRECT_ROUTES[t][2])} for t in present],
            "missing": [{"tool": t, "binary": DIRECT_ROUTES[t][2]} for t in missing],
        }, indent=2)
    else:
        C = _cli_colors()
        b, g, w, R = C['ACCENT_LINE'], C['TERMINAL_GRAY'], C['BRIGHT_WHITE'], C['RESET']
        total = len(tools_to_check)
        widest = max(len(m) for m in missing) if missing else 0
        rows = []
        if missing:
            rows.append(f"  {w}❌ Missing ({len(missing)}):{R}")
            for m in missing:
                rows.append(f"     {m:{widest}}  →  {g}{DIRECT_ROUTES[m][2]}{R}")
        if args.verbose and present:
            rows.append(f"  {w}✅ Present ({len(present)}):{R}")
            for p in present:
                rows.append(f"     {p:{widest}}  →  {g}{shutil.which(DIRECT_ROUTES[p][2])}{R}")
        inner = max(_dw(r) for r in rows) if rows else 30
        def p(s): return f"  {b}│{R}  {s}{' ' * (inner - _dw(s))}  {b}│{R}"
        buf = [f"  {b}╭{'─' * (inner + 4)}╮{R}"]
        buf.append(p(f"  {w}Validate:{R} {len(present)}/{total} tools available"))
        for r in rows:
            buf.append(p(r))
        buf.append(f"  {b}╰{'─' * (inner + 4)}╯{R}")
        output = '\n'.join(buf) + '\n'

    print(output, end="")
    if args.output:
        Path(args.output).write_text(output)


# ============================================================================
# Subcommand: mcp
# ============================================================================

def cmd_mcp(args):
    """Run MCP stdio bridge (Claude Desktop)."""
    from mcp_core.mcp_entry import run_mcp as _run_mcp

    class MockArgs:
        debug = args.debug
        server = args.server
        timeout = args.timeout
        compact = args.compact
        profile = args.profile
        auth_token = args.auth_token
        disable_ssl_verify = args.disable_ssl_verify

    _run_mcp(MockArgs(), logger)


# ============================================================================
# Subcommand: ctf
# ============================================================================

def cmd_ctf(args):
    """CTF challenge workflow analysis."""
    from server_core.workflows.ctf.CTFChallenge import CTFChallenge
    from server_core.workflows.ctf.automator import CTFWorkflowManager
    from server_core.workflows.ctf.CTFChallenge import CTFChallenge

    challenge = CTFChallenge(
        name=name,
        category=category,
        description=description,
        difficulty=difficulty,
        points=points,
        target=target,
    )

    wm = CTFWorkflowManager()
    result = wm.create_ctf_challenge_workflow(challenge)
    steps = result.get("workflow_steps", [])

    if args.json:
        output = json.dumps({
            "name": name,
            "category": category,
            "difficulty": difficulty,
            "points": points,
            "target": target,
            "total_steps": len(steps),
            "steps": steps,
        }, indent=2)
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            print(f"🏴 CTF: {name} [{category}, {difficulty}]")
            print(f"   Points: {points} | Target: {target or '(unknown)'}")
            print(f"\nWorkflow ({len(steps)} steps):")
            for i, step in enumerate(steps, 1):
                tool = step.get("action", step.get("step", "?"))
                desc = step.get("description", "")
                print(f"  {i:2d}. {tool:20s} {desc[:80]}")
        output = buf.getvalue()

    print(output, end="")
    if args.output:
        Path(args.output).write_text(output)


# ============================================================================
# CLI entry point
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 hexstrike.py",
        description="HexStrike AI-PULSE CLI — Cybersecurity automation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hexstrike.py serve                         Start server on :8888
  python3 hexstrike.py serve --host 0.0.0.0 --port 8080
  python3 hexstrike.py scan nmap target=scanme.nmap.org scan_type=-sV
  python3 hexstrike.py scan nmap target=scanme.nmap.org scan_type=-sV --json -o result.json
  python3 hexstrike.py scan nuclei target=http://scanme.nmap.org severity=critical
  python3 hexstrike.py tools --filter nmap
  python3 hexstrike.py status --host 192.168.1.10
  python3 hexstrike.py validate --verbose
  python3 hexstrike.py mcp --debug
  python3 hexstrike.py ctf --category pwn --difficulty hard
        """,
    )
    parser.add_argument("--version", action="version", version=f"hexstrike {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start Pulse HTTP/SSE server")
    p_serve.add_argument("--host", default=None, help=f"Bind address (default: {DEFAULT_HOST})")
    p_serve.add_argument("--port", type=int, default=None, help=f"Bind port (default: {DEFAULT_PORT})")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug logging")

    # scan
    p_scan = sub.add_parser("scan", help="Run a security tool directly")
    p_scan.add_argument("tool", help="Tool name (nmap, nuclei, whatweb, ...)")
    p_scan.add_argument("target", nargs="?", default="", help="Target (host/URL/domain)")
    p_scan.add_argument("-p", "--param", action="append", default=[], help="Extra params key=val")
    p_scan.add_argument("--json", action="store_true", help="Output raw JSON")
    p_scan.add_argument("-o", "--output", default="", help="Write output to file")

    # tools
    p_tools = sub.add_parser("tools", help="List available tools")
    p_tools.add_argument("--filter", "-f", default="", help="Filter tools by name")
    p_tools.add_argument("--json", action="store_true", help="Output raw JSON")
    p_tools.add_argument("-o", "--output", default="", help="Write output to file")

    # status
    p_status = sub.add_parser("status", help="Check server health")
    p_status.add_argument("--host", default=None, help=f"Server host (default: {DEFAULT_HOST})")
    p_status.add_argument("--port", type=int, default=None, help=f"Server port (default: {DEFAULT_PORT})")
    p_status.add_argument("--json", action="store_true", help="Output raw JSON")
    p_status.add_argument("-o", "--output", default="", help="Write output to file")

    # validate
    p_val = sub.add_parser("validate", help="Validate tool environment")
    p_val.add_argument("--tool-filter", "-f", default="", help="Filter specific tools")
    p_val.add_argument("--verbose", "-v", action="store_true", help="Show present tools too")
    p_val.add_argument("--json", action="store_true", help="Output raw JSON")
    p_val.add_argument("-o", "--output", default="", help="Write output to file")

    # mcp (stdio bridge)
    p_mcp = sub.add_parser("mcp", help="Run MCP stdio bridge (Claude Desktop)")
    p_mcp.add_argument("--server", default="http://127.0.0.1:8888", help="Pulse server URL")
    p_mcp.add_argument("--timeout", type=int, default=300, help="Request timeout (s)")
    p_mcp.add_argument("--debug", action="store_true", help="Enable debug logging")
    p_mcp.add_argument("--compact", action="store_true", help="Compact mode for small LLMs")
    p_mcp.add_argument("--profile", nargs="+", default=[], help="Tool profiles to load")
    p_mcp.add_argument("--auth-token", default="", help="Bearer token for auth")
    p_mcp.add_argument("--disable-ssl-verify", action="store_true", help="Disable SSL verification")

    # ctf
    p_ctf = sub.add_parser("ctf", help="CTF challenge workflow")
    p_ctf.add_argument("--category", default="web", help="Challenge category (web/pwn/crypto/forensics/re/misc)")
    p_ctf.add_argument("--name", default="", help="Challenge name")
    p_ctf.add_argument("--description", default="", help="Challenge description")
    p_ctf.add_argument("--difficulty", default="medium", help="Difficulty (easy/medium/hard/insane)")
    p_ctf.add_argument("--points", type=int, default=0, help="Challenge points")
    p_ctf.add_argument("--target", default="", help="Target IP/host")
    p_ctf.add_argument("--json", action="store_true", help="Output raw JSON")
    p_ctf.add_argument("-o", "--output", default="", help="Write output to file")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "tools":
        cmd_tools(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "ctf":
        cmd_ctf(args)


if __name__ == "__main__":
    main()
