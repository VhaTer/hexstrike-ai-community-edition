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
   hexstrike ctf        CTF challenge analysis with live nmap scan

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
from mcp_core.tool_routes import TOOL_ROUTES

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("hexstrike")

VERSION = "0.10.1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888


def _resolve_tool(tool_name: str):
    route = TOOL_ROUTES.get(tool_name)
    if not route:
        return None
    module_path, exec_name, tool_key = route
    mod = importlib.import_module(module_path)
    exec_func = getattr(mod, exec_name)
    return exec_func, tool_key


def _call_with_stdout_suppressed(enabled: bool, func, *args, **kwargs):
    if not enabled:
        return func(*args, **kwargs)
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


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


def _print_output(output: str, args):
    print(output, end="")
    if hasattr(args, 'output') and args.output:
        Path(args.output).write_text(output)


def _emit_json(data, args):
    output = json.dumps(data, indent=2, default=str)
    _print_output(output, args)


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
    logger.info(f"Starting HexStrike Pulse Server")
    logger.info(f"Server: http://{host}:{port}")

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

def _unknown_tool_json(tool_name):
    similar = [t for t in TOOL_ROUTES if tool_name in t or t.startswith(tool_name[:3])]
    return {"error": f"Unknown tool: {tool_name}", "similar": similar[:5], "success": False}

def cmd_scan(args):
    """Run a security tool directly (no server required)."""
    tool_name = args.tool

    resolved = _call_with_stdout_suppressed(args.json, _resolve_tool, tool_name)
    if not resolved:
        if args.json:
            _emit_json(_unknown_tool_json(tool_name), args)
        else:
            logger.error(f"Unknown tool: {tool_name}")
            similar = [t for t in TOOL_ROUTES if tool_name in t or t.startswith(tool_name[:3])]
            if similar:
                logger.info(f"   Did you mean: {', '.join(similar[:5])}?")
        sys.exit(1)

    exec_func, tool_key = resolved

    params = {}
    target = args.target
    if target:
        params["target"] = target
        params["url"] = target
    for p in args.param:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k] = v

    result = _call_with_stdout_suppressed(args.json, exec_func, tool_key, params)

    if args.json:
        _emit_json(result, args)
    else:
        output = _format_scan_result(tool_name, target, result)
        _print_output(output, args)


def _format_scan_result(tool_name: str, target: str, result: dict) -> str:
    C = _cli_colors()
    b, g, w, R = C['ACCENT_LINE'], C['TERMINAL_GRAY'], C['BRIGHT_WHITE'], C['RESET']
    out = result.get("output", "") or result.get("stdout", "")
    err = result.get("error", "") or result.get("stderr", "")
    success = result.get("success", False)
    timed_out = result.get("timed_out", False)
    exec_time = result.get("execution_time", 0)
    if success:
        icon, sl = "\u2705", f"Success ({exec_time:.1f}s)"
    elif timed_out:
        icon, sl = "\u23f0", f"Timed out ({exec_time:.1f}s)"
    else:
        icon, sl = "\u274c", "Failed"
    rows = [
        f"  {w}Tool:{R}    {tool_name}",
        f"  {w}Target:{R}  {target or '(none)'}",
        f"  {w}Status:{R}  {icon} {sl}",
    ]
    inner = max(_dw(r) for r in rows)
    def p(s): return f"  {b}\u2502{R}  {s}{' ' * (inner - _dw(s))}  {b}\u2502{R}"
    buf = [f"  {b}\u250c{'\u2500' * (inner + 4)}\u2510{R}"]
    for r in rows:
        buf.append(p(r))
    buf.append(f"  {b}\u2514{'\u2500' * (inner + 4)}\u2518{R}")
    body = out.strip() if out else ""
    if err:
        body += f"\n{g}Error: {err.strip()[:500]}{R}" if body else f"{g}Error: {err.strip()[:500]}{R}"
    if body:
        buf.append("")
        buf.append(body)
    return '\n'.join(buf) + '\n'


# ============================================================================
# Subcommand: tools
# ============================================================================

def cmd_tools(args):
    """List available tools with their descriptions."""
    from tool_registry import TOOLS

    if args.filter:
        matching = {k: v for k, v in TOOLS.items() if args.filter.lower() in k.lower()}
        if not matching:
            if args.json:
                _emit_json({}, args)
            else:
                print(f"No tools matching '{args.filter}'")
                if args.output:
                    Path(args.output).write_text(f"No tools matching '{args.filter}'\n")
            return
        source = matching
    else:
        source = TOOLS

    cat_order = ["essential", "recon", "web_vuln", "web_fuzz", "password",
                 "smb", "exploit", "osint", "cloud", "binary", "wifi", "misc"]

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
        _emit_json(data, args)
    else:
        buf = []
        idx = 0
        for cat in cat_order:
            if cat not in by_cat:
                continue
            buf.append(f"\n\u2500\u2500 {cat} \u2500\u2500")
            for name, defn in sorted(by_cat[cat]):
                idx += 1
                desc = defn.get("desc", "")
                eff = defn.get("effectiveness", "")
                eff_str = f" [{eff:.0%}]" if isinstance(eff, (int, float)) else ""
                buf.append(f"({idx:2d}) {name:22s} {desc}{eff_str}")
        for cat in sorted(set(by_cat) - set(cat_order)):
            buf.append(f"\n\u2500\u2500 {cat} \u2500\u2500")
            for name, defn in sorted(by_cat[cat]):
                idx += 1
                buf.append(f"({idx:2d}) {name:22s} {defn.get('desc', '')}")
        output = '\n'.join(buf) + '\n'
        _print_output(output, args)


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

        if args.json:
            _emit_json({
                "status": status, "uptime_seconds": uptime,
                "host": host, "port": port, "checks": checks,
            }, args)
        else:
            C = _cli_colors()
            b, g, w, R = C['ACCENT_LINE'], C['TERMINAL_GRAY'], C['BRIGHT_WHITE'], C['RESET']
            icon = "\U0001f7e2" if status == "ready" else "\U0001f7e1" if status == "degraded" else "\U0001f534"
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
            def p(s): return f"  {b}\u2502{R}  {s}{' ' * (inner - _dw(s))}  {b}\u2502{R}"
            buf = [f"  {b}\u250c{'\u2500' * (inner + 4)}\u2510{R}"]
            buf.append(p(rows[0]))
            buf.extend(p(r) for r in rows[1:-1])
            buf.append(f"  {b}\u2502{R}  {g}{url}{R}{' ' * (inner - len(url))}  {b}\u2502{R}")
            buf.append(f"  {b}\u2514{'\u2500' * (inner + 4)}\u2518{R}")
            _print_output('\n'.join(buf) + '\n', args)
    except urllib.error.URLError:
        if args.json:
            _emit_json({"error": f"Server not responding at {url}", "success": False}, args)
        else:
            _print_output(f"\U0001f534 Pulse server not responding at {url}\n", args)
        sys.exit(1)
    except Exception as e:
        if args.json:
            _emit_json({"error": str(e), "success": False}, args)
        else:
            _print_output(f"\U0001f534 Error: {e}\n", args)
        sys.exit(1)


# ============================================================================
# Subcommand: validate
# ============================================================================

def cmd_validate(args):
    """Validate which external tools are available."""
    import shutil

    tools_to_check = list(TOOL_ROUTES.keys())
    if args.tool_filter:
        tools_to_check = [t for t in tools_to_check if args.tool_filter.lower() in t.lower()]

    present = []
    missing = []
    for tool_name in sorted(tools_to_check):
        _, _, binary_key = TOOL_ROUTES[tool_name]
        path = shutil.which(binary_key)
        if path:
            present.append(tool_name)
        else:
            missing.append(tool_name)

    if args.json:
        _emit_json({
            "total": len(tools_to_check),
            "present_count": len(present),
            "missing_count": len(missing),
            "present": [{"tool": t, "binary": TOOL_ROUTES[t][2], "path": shutil.which(TOOL_ROUTES[t][2])} for t in present],
            "missing": [{"tool": t, "binary": TOOL_ROUTES[t][2]} for t in missing],
        }, args)
    else:
        C = _cli_colors()
        b, g, w, R = C['ACCENT_LINE'], C['TERMINAL_GRAY'], C['BRIGHT_WHITE'], C['RESET']
        total = len(tools_to_check)
        widest = max(len(m) for m in missing) if missing else 0
        rows = []
        if missing:
            rows.append(f"  {w}Missing ({len(missing)}):{R}")
            for m in missing:
                rows.append(f"     {m:{widest}}  \u2192  {g}{TOOL_ROUTES[m][2]}{R}")
        if args.verbose and present:
            rows.append(f"  {w}Present ({len(present)}):{R}")
            for p in present:
                rows.append(f"     {p:{widest}}  \u2192  {g}{shutil.which(TOOL_ROUTES[p][2])}{R}")
        inner = max(_dw(r) for r in rows) if rows else 30
        def p(s): return f"  {b}\u2502{R}  {s}{' ' * (inner - _dw(s))}  {b}\u2502{R}"
        buf = [f"  {b}\u250c{'\u2500' * (inner + 4)}\u2510{R}"]
        buf.append(p(f"  {w}Validate:{R} {len(present)}/{total} tools available"))
        for r in rows:
            buf.append(p(r))
        buf.append(f"  {b}\u2514{'\u2500' * (inner + 4)}\u2518{R}")
        _print_output('\n'.join(buf) + '\n', args)


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
    """CTF challenge workflow analysis with real scans when target is provided."""
    category = args.category
    name = args.name or f"{category.upper()} Challenge"
    description = args.description or f"{category} CTF challenge"
    difficulty = args.difficulty
    points = args.points
    target = args.target

    # Build static workflow plan
    def build_workflow():
        from server_core.workflows.ctf.CTFChallenge import CTFChallenge
        from server_core.workflows.ctf.workflowManager import CTFWorkflowManager
        challenge = CTFChallenge(
            name=name, category=category, description=description,
            difficulty=difficulty, points=points, target=target or None,
        )
        wm = CTFWorkflowManager()
        return wm.create_ctf_challenge_workflow(challenge)

    result = _call_with_stdout_suppressed(args.json, build_workflow)
    steps = result.get("workflow_steps", [])

    # Real scans when target provided (fast tools only, 30s timeout each)
    scan_results = []
    if target:
        for scan_tool, scan_params, scan_label in _ctf_scans_for(category, target):
            resolved = _resolve_tool(scan_tool)
            if not resolved:
                continue
            exec_func, tool_key = resolved
            try:
                with redirect_stdout(io.StringIO()):
                    scan_out = exec_func(tool_key, scan_params)
                scan_results.append({"tool": scan_label, "result": scan_out})
            except Exception as e:
                scan_results.append({"tool": scan_label, "result": {"success": False, "error": str(e)}})

    if args.json:
        output = {
            "name": name, "category": category, "difficulty": difficulty,
            "points": points, "target": target or "",
            "total_steps": len(steps), "steps": steps,
            "tools": result.get("tools", []),
            "estimated_time": result.get("estimated_time", 0),
            "success_probability": result.get("success_probability", 0),
        }
        if scan_results:
            output["scans"] = scan_results
        _emit_json(output, args)
    else:
        buf = [f"CTF: {name} [{category}, {difficulty}]",
               f"   Points: {points} | Target: {target or '(unknown)'}"]
        tools = result.get("tools", [])
        if tools:
            buf.append(f"\nRecommended tools ({len(tools)}):")
            for t in tools:
                buf.append(f"  - {t}")
        est = result.get("estimated_time", 0)
        prob = result.get("success_probability", 0)
        buf.append(f"\nEst. time: {est//60}m | Success rate: {prob:.0%}")
        buf.append(f"\nWorkflow ({len(steps)} steps):")
        for i, step in enumerate(steps, 1):
            action = step.get("action", step.get("step", "?"))
            desc = step.get("description", "")
            buf.append(f"  {i:2d}. {action:20s} {desc[:80]}")

        if scan_results:
            buf.append(f"\nLive scans ({len(scan_results)}):")
            for s in scan_results:
                out = s["result"]
                success = out.get("success", False)
                err = out.get("error", "") or ""
                scan_out = (out.get("output", "") or out.get("stdout", "") or "").strip()
                lines = scan_out.splitlines() if scan_out else []
                if not lines and err:
                    lines = [f"error: {err[:120]}"]
                buf.append(f"  [{s['tool']}] {'OK' if success else 'FAIL'}")
                for line in lines[:5]:
                    buf.append(f"    {line}")
                if len(lines) > 5:
                    buf.append(f"    ... ({len(lines) - 5} more lines)")
        buf.append("")
        _print_output('\n'.join(buf), args)


def _ctf_scans_for(category: str, target: str) -> list[tuple[str, dict, str]]:
    """Return (tool, params, label) tuples for quick CTF target scans.

    Fast scans only (seconds, not minutes). Heavy scans (nuclei, whatweb)
    are left to 'hexstrike scan <tool> <target>'.
    """
    from urllib.parse import urlparse

    scans: list[tuple[str, dict, str]] = []

    # Extract hostname from URL for nmap
    if target.startswith(("http://", "https://")):
        host = urlparse(target).hostname
        if not host:
            return scans
        nmap_target = host
    else:
        nmap_target = target

    if category in ("web", "misc"):
        scans.append(("nmap", {"target": nmap_target, "scan_type": "-sT -Pn -T4", "ports": "22,80,443,8080"}, "nmap"))
    elif category in ("pwn", "rev"):
        scans.append(("nmap", {"target": nmap_target, "scan_type": "-sT -Pn -T4 -sV", "ports": "22,80,443,4444,1337,31337"}, "nmap"))
    else:
        scans.append(("nmap", {"target": nmap_target, "scan_type": "-sT -Pn -T4", "ports": "22,80,443"}, "nmap"))

    return scans





# ============================================================================
# CLI entry point
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexstrike",
        description="HexStrike AI-PULSE CLI \u2014 Cybersecurity automation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hexstrike serve                         Start server on :8888
  hexstrike serve --host 0.0.0.0 --port 8080
  hexstrike scan nmap scanme.nmap.org -p scan_type=-sV
  hexstrike scan nmap scanme.nmap.org --json -o result.json
  hexstrike scan nuclei http://scanme.nmap.org -p severity=critical
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

    # mcp
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

    commands = {
        "serve":    cmd_serve,
        "scan":     cmd_scan,
        "tools":    cmd_tools,
        "status":   cmd_status,
        "validate": cmd_validate,
        "mcp":      cmd_mcp,
        "ctf":      cmd_ctf,
    }

    cmd = commands.get(args.command)
    if cmd:
        try:
            cmd(args)
        except Exception as e:
            if getattr(args, 'json', False):
                _emit_json({"error": f"{args.command} failed: {e}", "success": False}, args)
            else:
                logger.error(f"{args.command} failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
