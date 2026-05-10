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
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("hexstrike")

VERSION = "1.0.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888

# ---------------------------------------------------------------------------
# Direct tool routing (same pattern as DIRECT_TOOLS in server_setup.py)
# ---------------------------------------------------------------------------
DIRECT_ROUTES = {
    "amass":         ("mcp_core.recon_direct", "recon_exec", "amass"),
    "subfinder":     ("mcp_core.recon_direct", "recon_exec", "subfinder"),
    "autorecon":     ("mcp_core.recon_direct", "recon_exec", "autorecon"),
    "theharvester":  ("mcp_core.recon_direct", "recon_exec", "theharvester"),
    "dnsenum":       ("mcp_core.recon_direct", "recon_exec", "dnsenum"),
    "fierce":        ("mcp_core.recon_direct", "recon_exec", "fierce"),
    "whois":         ("mcp_core.recon_direct", "recon_exec", "whois"),
    "nmap":          ("mcp_core.net_scan_direct", "net_scan_exec", "nmap"),
    "nmap_advanced": ("mcp_core.net_scan_direct", "net_scan_exec", "nmap-advanced"),
    "masscan":       ("mcp_core.net_scan_direct", "net_scan_exec", "masscan"),
    "rustscan":      ("mcp_core.net_scan_direct", "net_scan_exec", "rustscan"),
    "arp_scan":      ("mcp_core.net_scan_direct", "net_scan_exec", "arp-scan"),
    "nikto":         ("mcp_core.web_scan_direct", "web_scan_exec", "nikto"),
    "sqlmap":        ("mcp_core.web_scan_direct", "web_scan_exec", "sqlmap"),
    "wpscan":        ("mcp_core.web_scan_direct", "web_scan_exec", "wpscan"),
    "dalfox":        ("mcp_core.web_scan_direct", "web_scan_exec", "dalfox"),
    "xsser":         ("mcp_core.web_scan_direct", "web_scan_exec", "xsser"),
    "gobuster":      ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "gobuster"),
    "ffuf":          ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "ffuf"),
    "feroxbuster":   ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "feroxbuster"),
    "dirsearch":     ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "dirsearch"),
    "dirb":          ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "dirb"),
    "wfuzz":         ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "wfuzz"),
    "dotdotpwn":     ("mcp_core.web_fuzz_direct", "web_fuzz_exec", "dotdotpwn"),
    "hydra":         ("mcp_core.password_cracking_direct", "pwdcrack_exec", "hydra"),
    "hashcat":       ("mcp_core.password_cracking_direct", "pwdcrack_exec", "hashcat"),
    "john":          ("mcp_core.password_cracking_direct", "pwdcrack_exec", "john"),
    "medusa":        ("mcp_core.password_cracking_direct", "pwdcrack_exec", "medusa"),
    "patator":       ("mcp_core.password_cracking_direct", "pwdcrack_exec", "patator"),
    "hashid":        ("mcp_core.password_cracking_direct", "pwdcrack_exec", "hashid"),
    "enum4linux":    ("mcp_core.smb_enum_direct", "smb_enum_exec", "enum4linux"),
    "netexec":       ("mcp_core.smb_enum_direct", "smb_enum_exec", "netexec"),
    "rpcclient":     ("mcp_core.smb_enum_direct", "smb_enum_exec", "rpcclient"),
    "smbmap":        ("mcp_core.smb_enum_direct", "smb_enum_exec", "smbmap"),
    "nbtscan":       ("mcp_core.smb_enum_direct", "smb_enum_exec", "nbtscan"),
    "metasploit":    ("mcp_core.exploit_framework_direct", "exploit_exec", "metasploit"),
    "msfvenom":      ("mcp_core.exploit_framework_direct", "exploit_exec", "msfvenom"),
    "searchsploit":  ("mcp_core.exploit_framework_direct", "exploit_exec", "exploit_db"),
    "exploit_db":    ("mcp_core.exploit_framework_direct", "exploit_exec", "exploit_db"),
    "katana":        ("mcp_core.web_recon_direct", "web_recon_exec", "katana"),
    "hakrawler":     ("mcp_core.web_recon_direct", "web_recon_exec", "hakrawler"),
    "gau":           ("mcp_core.web_recon_direct", "web_recon_exec", "gau"),
    "waybackurls":   ("mcp_core.web_recon_direct", "web_recon_exec", "waybackurls"),
    "httpx":         ("mcp_core.web_recon_direct", "web_recon_exec", "httpx"),
    "wafw00f":       ("mcp_core.web_recon_direct", "web_recon_exec", "wafw00f"),
    "arjun":         ("mcp_core.web_recon_direct", "web_recon_exec", "arjun"),
    "paramspider":   ("mcp_core.web_recon_direct", "web_recon_exec", "paramspider"),
    "x8":            ("mcp_core.web_recon_direct", "web_recon_exec", "x8"),
    "prowler":       ("mcp_core.security_direct", "security_exec", "prowler"),
    "trivy":         ("mcp_core.security_direct", "security_exec", "trivy"),
    "kube_hunter":   ("mcp_core.security_direct", "security_exec", "kube-hunter"),
    "kube_bench":    ("mcp_core.security_direct", "security_exec", "kube-bench"),
    "checkov":       ("mcp_core.security_direct", "security_exec", "checkov"),
    "terrascan":     ("mcp_core.security_direct", "security_exec", "terrascan"),
    "ropgadget":     ("mcp_core.misc_direct", "misc_exec", "ropgadget"),
    "ropper":        ("mcp_core.misc_direct", "misc_exec", "ropper"),
    "one_gadget":    ("mcp_core.misc_direct", "misc_exec", "one_gadget"),
    "volatility":    ("mcp_core.misc_direct", "misc_exec", "volatility"),
    "volatility3":   ("mcp_core.misc_direct", "misc_exec", "volatility3"),
    "gdb":           ("mcp_core.misc_direct", "misc_exec", "gdb"),
    "radare2":       ("mcp_core.misc_direct", "misc_exec", "radare2"),
    "strings":       ("mcp_core.misc_direct", "misc_exec", "strings"),
    "objdump":       ("mcp_core.misc_direct", "misc_exec", "objdump"),
    "checksec":      ("mcp_core.misc_direct", "misc_exec", "checksec"),
    "binwalk":       ("mcp_core.misc_direct", "misc_exec", "binwalk"),
    "steghide":      ("mcp_core.misc_direct", "misc_exec", "steghide"),
    "foremost":      ("mcp_core.misc_direct", "misc_exec", "foremost"),
    "exiftool":      ("mcp_core.misc_direct", "misc_exec", "exiftool"),
    "mysql":         ("mcp_core.misc_direct", "misc_exec", "mysql"),
    "sqlite":        ("mcp_core.misc_direct", "misc_exec", "sqlite"),
    "nuclei":        ("mcp_core.misc_direct", "misc_exec", "nuclei"),
    "testssl":       ("mcp_core.testssl_direct", "testssl_exec", "testssl"),
    "whatweb":       ("mcp_core.web_probe_direct", "web_probe_exec", "whatweb"),
    "commix":        ("mcp_core.web_probe_direct", "web_probe_exec", "commix"),
    "joomscan":      ("mcp_core.web_probe_direct", "web_probe_exec", "joomscan"),
    "vulnx":         ("mcp_core.vuln_intel_direct", "vuln_intel_exec", "vulnx"),
    "airmon_ng":     ("mcp_core.wifi_direct", "wifi_exec", "airmon_ng"),
    "airodump_ng":   ("mcp_core.wifi_direct", "wifi_exec", "airodump_ng"),
    "aireplay_ng":   ("mcp_core.wifi_direct", "wifi_exec", "aireplay_ng"),
    "aircrack_ng":   ("mcp_core.wifi_direct", "wifi_exec", "aircrack_ng"),
    "hcxdumptool":   ("mcp_core.wifi_direct", "wifi_exec", "hcxdumptool"),
    "wifite":        ("mcp_core.wifi_direct", "wifi_exec", "wifite2"),
    "mdk4":          ("mcp_core.wifi_direct", "wifi_exec", "mdk4"),
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


def _format_result(result: dict) -> str:
    """Pretty-print a tool result dict."""
    out = result.get("output", "") or result.get("stdout", "")
    err = result.get("error", "") or result.get("stderr", "")
    success = result.get("success", False)
    lines = []
    if success:
        lines.append(f"✅ Success ({result.get('execution_time', 0):.1f}s)")
    else:
        timed_out = result.get("timed_out", False)
        if timed_out:
            lines.append(f"⏰ Timed out ({result.get('execution_time', 0):.1f}s)")
        else:
            lines.append(f"❌ Failed")
    if out:
        lines.append(out.strip()[:2000])
    if err:
        lines.append(f"Error: {err.strip()[:500]}")
    return "\n".join(lines)


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

    # Build params from CLI args
    params = {}
    if target:
        # Most tools use target; some use domain, url, etc.
        params["target"] = target
        params["url"] = target
    if args.param:
        for p in args.param:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v

    logger.info(f"🔍 Running {tool_name} against {target or '...'}")
    result = exec_func(tool_key, params)
    print(_format_result(result))


# ============================================================================
# Subcommand: tools
# ============================================================================

def cmd_tools(args):
    """List available tools with their descriptions."""
    from tool_registry import TOOLS

    if args.filter:
        matching = {k: v for k, v in TOOLS.items() if args.filter.lower() in k.lower()}
        if not matching:
            logger.info(f"No tools matching '{args.filter}'")
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

    for cat in cat_order:
        if cat not in by_cat:
            continue
        print(f"\n── {cat} ──")
        for name, defn in sorted(by_cat[cat]):
            desc = defn.get("desc", "")
            eff = defn.get("effectiveness", "")
            eff_str = f" [{eff:.0%}]" if isinstance(eff, (int, float)) else ""
            print(f"  {name:20s} {desc}{eff_str}")

    for cat in sorted(set(by_cat) - set(cat_order)):
        print(f"\n── {cat} ──")
        for name, defn in sorted(by_cat[cat]):
            print(f"  {name:20s} {defn.get('desc', '')}")


# ============================================================================
# Subcommand: status
# ============================================================================

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

        icon = "🟢" if status == "ready" else "🟡" if status == "degraded" else "🔴"
        print(f"{icon} Pulse server: {status}")
        print(f"   Uptime: {uptime // 3600}h{uptime % 3600 // 60}m")
        tools = checks.get("essential_tools", {})
        print(f"   Tools: {tools.get('available', '?')}/{tools.get('total', '?')} essential")
        disk = checks.get("disk", {})
        if disk:
            print(f"   Disk: {disk.get('free_gb', '?')} GB free ({disk.get('usage_pct', '?')}% used)")
        print(f"   URL: {url}")
    except urllib.error.URLError:
        print(f"🔴 Pulse server not responding at {url}")
        sys.exit(1)
    except Exception as e:
        print(f"🔴 Error: {e}")
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

    print(f"✅ {len(present)} present   ❌ {len(missing)} missing   Total: {len(tools_to_check)}")
    if missing:
        print(f"\nMissing ({len(missing)}):")
        for m in missing:
            print(f"  {m}: {DIRECT_ROUTES[m][2]}")
    if args.verbose and present:
        print(f"\nPresent ({len(present)}):")
        for p in present:
            print(f"  {p}: {shutil.which(DIRECT_ROUTES[p][2])}")


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
    """CTF challenge analysis."""
    category = args.category or "web"
    name = args.name or f"CTF-{category}-challenge"
    description = args.description or f"A {category} CTF challenge"
    difficulty = args.difficulty or "medium"
    points = args.points or 0
    target = args.target or ""

    from server_core.workflows.ctf.workflowManager import CTFWorkflowManager
    from shared.target_types import TargetType
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

    print(f"🏴 CTF: {name} [{category}, {difficulty}]")
    print(f"   Points: {points} | Target: {target or '(unknown)'}")
    print(f"\nWorkflow ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        tool = step.get("action", step.get("step", "?"))
        desc = step.get("description", "")
        print(f"  {i:2d}. {tool:20s} {desc[:80]}")


# ============================================================================
# CLI entry point
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexstrike",
        description="HexStrike AI-PULSE CLI — Cybersecurity automation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hexstrike serve                         Start server on :8888
  hexstrike serve --host 0.0.0.0 --port 8080
  hexstrike scan nmap target=scanme.nmap.org scan_type=-sV
  hexstrike scan nuclei target=http://scanme.nmap.org severity=critical
  hexstrike tools --filter nmap
  hexstrike status --host 192.168.1.10
  hexstrike validate --verbose
  hexstrike mcp --debug
  hexstrike ctf --category pwn --difficulty hard
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

    # tools
    p_tools = sub.add_parser("tools", help="List available tools")
    p_tools.add_argument("--filter", "-f", default="", help="Filter tools by name")

    # status
    p_status = sub.add_parser("status", help="Check server health")
    p_status.add_argument("--host", default=None, help=f"Server host (default: {DEFAULT_HOST})")
    p_status.add_argument("--port", type=int, default=None, help=f"Server port (default: {DEFAULT_PORT})")

    # validate
    p_val = sub.add_parser("validate", help="Validate tool environment")
    p_val.add_argument("--tool-filter", "-f", default="", help="Filter specific tools")
    p_val.add_argument("--verbose", "-v", action="store_true", help="Show present tools too")

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
