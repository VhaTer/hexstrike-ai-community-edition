#!/usr/bin/env python3
"""
HexStrike AI-PULSE — FastMCP Standalone Server

🚀 Direct tool execution, native HTTP/SSE transport
📊 Dashboard served from server_static/ via Starlette StaticFiles
"""

import os
import sys
import argparse
import logging
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime

from starlette.responses import FileResponse, Response, StreamingResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from mcp_core.server_setup import setup_mcp_server_standalone, _op_metrics, _scan_cache
from server_core.modern_visual_engine import ModernVisualEngine
from server_core.singletons import cache, telemetry, enhanced_process_manager, error_handler
import server_core.config_core as config_core

# ---------------------------------------------------------------------------
# Tool availability (formerly server_api/ops/system_monitoring.py)
# ---------------------------------------------------------------------------
import shutil

_HEALTH_TOOL_CATEGORIES = {
    "essential": ["nmap", "curl", "python3"],
    "recon":     ["subfinder", "amass", "httpx", "katana"],
    "web":       ["nikto", "sqlmap", "gobuster", "ffuf", "nuclei"],
    "wifi":      ["airmon-ng", "airodump-ng", "aircrack-ng"],
    "exploit":   ["msfconsole", "searchsploit"],
}
_tool_availability_last_refresh: float = 0.0
_tool_availability_cache: dict = {}


def _get_tool_availability() -> dict:
    """Check which pentest tools are available on PATH."""
    global _tool_availability_last_refresh, _tool_availability_cache
    import time
    if time.time() - _tool_availability_last_refresh < 60 and _tool_availability_cache:
        return _tool_availability_cache
    all_tools = {t for tools in _HEALTH_TOOL_CATEGORIES.values() for t in tools}
    result = {tool: shutil.which(tool) is not None for tool in all_tools}
    _tool_availability_cache = result
    _tool_availability_last_refresh = time.time()
    return result

# ============================================================================
# LOGGING CONFIGURATION (MUST BE FIRST)
# ============================================================================

from server_core.setup_logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Configuration
API_HOST = os.environ.get('HEXSTRIKE_HOST', '127.0.0.1')
API_PORT = int(os.environ.get('HEXSTRIKE_PORT', 8888))


def _build_dashboard_response():
    """Build the web dashboard response data."""
    try:
        tools_status = _get_tool_availability()
        essential_tools = _HEALTH_TOOL_CATEGORIES.get("essential", [])
        all_essential_available = all(tools_status.get(t, False) for t in essential_tools)

        category_stats = {
            cat: {
                "total": len(tools),
                "available": sum(1 for t in tools if tools_status.get(t, False)),
            }
            for cat, tools in _HEALTH_TOOL_CATEGORIES.items()
        }

        current_usage = enhanced_process_manager.resource_monitor.get_current_usage()

        return {
            # Server identity
            "status": "healthy" if all_essential_available else "degraded",
            "version": config_core.get("VERSION", "unknown"),
            "uptime": time.time() - telemetry.stats.get("start_time", time.time()),

            # Telemetry / commands
            "telemetry": telemetry.get_stats(),

            # Tool availability
            "tools_status": tools_status,
            "all_essential_tools_available": all_essential_available,
            "total_tools_available": sum(1 for v in tools_status.values() if v),
            "total_tools_count": len(tools_status),
            "category_stats": category_stats,
            "category_tools": _HEALTH_TOOL_CATEGORIES,
            "tool_availability_age_seconds": round(time.time() - _tool_availability_last_refresh, 1) if _tool_availability_last_refresh > 0 else None,

            # System resources
            "resources": current_usage,
            "resources_timestamp": datetime.now().isoformat(),

            # Cache stats
            "cache_stats": cache.get_stats(),

            # Operational metrics (success rates, slowest tools, etc.)
            "op_metrics": _op_metrics.summary(),

            # Error statistics
            "error_stats": error_handler.get_error_statistics(),

            # Recent scan cache entries (last 10)
            "recent_scans": [
                {
                    "tool":           v.get("tool", "?"),
                    "target":         v.get("target", "?"),
                    "timestamp":      v.get("timestamp", 0),
                    "success":        v.get("result", {}).get("success", False),
                    "execution_time": v.get("result", {}).get("execution_time", 0),
                }
                for k, v in (list(_scan_cache.items())[-10:])
            ],
        }
    except Exception as e:
        logger.error(f"Error building web dashboard response: {e}")
        return {"error": f"Server error: {str(e)}", "status": "error"}


def _json_status_response(data):
    """Return JSON with HTTP status aligned to the embedded health state."""
    status = data.get("status")
    status_code = 200 if status == "healthy" else 500
    return JSONResponse(data, status_code=status_code)


def register_http_routes(mcp, logger, static_dir=None):
    """Attach dashboard and health HTTP routes to the active FastMCP server."""
    static_dir = Path(static_dir) if static_dir is not None else Path(__file__).parent / "server_static"

    if static_dir.exists():
        assets_dir = static_dir / "assets"

        # Mount /assets — Starlette StaticFiles handles MIME types correctly
        if assets_dir.exists():
            mcp._additional_http_routes.append(
                __import__("starlette.routing", fromlist=["Mount"]).Mount(
                    "/assets",
                    app=StaticFiles(directory=str(assets_dir)),
                    name="assets",
                )
            )

        @mcp.custom_route("/dashboard", methods=["GET"])
        async def dashboard_index(request):
            """Serve dashboard SPA index."""
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return Response("Dashboard not found", status_code=404)

        logger.info(f"📊 Dashboard at http://{API_HOST}:{API_PORT}/dashboard")
    else:
        logger.warning("⚠️ server_static/ not found — dashboard not available")

    @mcp.custom_route("/ping", methods=["GET"])
    async def ping(request):
        """Minimal liveness probe — is the process alive?"""
        return JSONResponse({"status": "ok", "server": "hexstrike-ai-pulse"})

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        """Readiness probe — server healthy and able to serve requests?
        
        Returns 200 when ready, 503 when degraded, 500 on error.
        """
        try:
            tools_status = _get_tool_availability()
            essential_tools = _HEALTH_TOOL_CATEGORIES.get("essential", [])
            all_essential_available = all(tools_status.get(t, False) for t in essential_tools)

            disk = shutil.disk_usage("/")
            disk_ok = disk.free / disk.total > 0.1

            ready = all_essential_available and disk_ok

            uptime = time.time() - telemetry.stats.get("start_time", time.time())

            return JSONResponse({
                "status": "ready" if ready else "degraded",
                "server": "hexstrike-ai-pulse",
                "version": config_core.get("VERSION", "unknown"),
                "uptime_seconds": int(uptime),
                "checks": {
                    "essential_tools": {
                        "status": "ok" if all_essential_available else "degraded",
                        "available": sum(1 for t in essential_tools if tools_status.get(t, False)),
                        "total": len(essential_tools),
                    },
                    "disk": {
                        "status": "ok" if disk_ok else "degraded",
                        "free_gb": round(disk.free / (1024**3), 1),
                        "total_gb": round(disk.total / (1024**3), 1),
                        "usage_pct": round((1 - disk.free / disk.total) * 100, 1),
                    },
                    "tools_available": sum(1 for v in tools_status.values() if v),
                    "tools_total": len(tools_status),
                },
            }, status_code=200 if ready else 503)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse({
                "status": "error",
                "server": "hexstrike-ai-pulse",
                "error": str(e)[:200],
            }, status_code=500)

    @mcp.custom_route("/web-dashboard", methods=["GET"])
    async def web_dashboard(request):
        """Combined endpoint for the web dashboard UI."""
        try:
            data = _build_dashboard_response()
            return JSONResponse(data)
        except Exception as e:
            logger.error(f"Error in /web-dashboard: {e}")
            return JSONResponse({"error": str(e), "status": "error"}, status_code=500)

    @mcp.custom_route("/web-dashboard/stream", methods=["GET"])
    async def stream_dashboard(request):
        """SSE endpoint — streams dashboard state."""
        async def generate():
            last_json = None
            while True:
                try:
                    dashboard = _build_dashboard_response()
                    js = json.dumps(dashboard, separators=(",", ":"))
                    if js != last_json:
                        yield f"data: {js}\n\n".encode()
                        last_json = js
                    else:
                        yield b": keepalive\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
                await asyncio.sleep(2)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    @mcp.custom_route("/api/logs/stream", methods=["GET"])
    async def stream_logs(request):
        """SSE endpoint — streams server logs to the dashboard Logs tab.

        Query params:
            lines (int): Number of recent lines to send initially (default 150).
        """
        from server_core.setup_logging import get_log_buffer
        buffer = get_log_buffer()

        try:
            n = int(request.query_params.get("lines", "150"))
        except (ValueError, TypeError):
            n = 150
        n = max(10, min(n, 1000))

        async def generate():
            q = buffer.subscribe()
            try:
                for line in buffer.get_recent(n):
                    yield f"data: {line}\n\n".encode()
                while True:
                    try:
                        line = await asyncio.wait_for(q.get(), timeout=5)
                        yield f"data: {line}\n\n".encode()
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                buffer.unsubscribe(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    if static_dir.exists():
        @mcp.custom_route("/{filename:str}", methods=["GET"])
        async def root_static(request):
            """Serve root static files (favicon, icons)."""
            filename = request.path_params.get("filename", "")
            file_path = static_dir / filename
            if file_path.is_file() and file_path.suffix in (".svg", ".ico", ".png", ".txt"):
                return FileResponse(str(file_path))
            return Response("Not found", status_code=404)

    logger.info("✅ Web dashboard endpoints registered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python3 hexstrike_server.py",
        description="HexStrike AI-PULSE — FastMCP Standalone Server",
    )
    parser.add_argument("--version", action="version", version=f"hexstrike_server {config_core.get('VERSION', '0.8.0')}")
    parser.add_argument("--host", default=None, help=f"Bind address (default: {API_HOST})")
    parser.add_argument("--port", type=int, default=None, help=f"Bind port (default: {API_PORT})")
    args = parser.parse_args()

    host = args.host or API_HOST
    port = args.port or API_PORT

    logging.getLogger("fastmcp").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    ver = config_core.get("VERSION", "0.8.0")
    print()
    print(ModernVisualEngine.create_banner(
        host=host, port=port, version=ver,
    ))
    from server_core.modern_visual_engine import ModernVisualEngine as _MVE
    from wcwidth import wcswidth
    import re as _re
    _asi = _re.compile(r'\x1b\[[0-9;]*m').sub
    C = _MVE.COLORS
    b = C['ACCENT_LINE']
    g = C['TERMINAL_GRAY']
    w = C['BRIGHT_WHITE']
    R = C['RESET']

    entries = [
        ("status",        "server health, uptime"),
        ("validate",      "check installed tools"),
        ("scan",          "run tools directly"),
        ("ctf",           "CTF attack planner"),
    ]
    rows = [f"  {g}${R} {w}python3 hexstrike.py {cmd:<12}{R} {g}·{R}  {desc}" for cmd, desc in entries]
    tip  = f"  {g}Tip:{R} add {w}-h/--help{R} to any subcommand for options"
    label = f"  {b}Useful HexStrike Pulse Commands{R}"
    inner = max(wcswidth(_asi('', s)) for s in [label, tip] + rows)

    banner_w = 76
    indent = max(0, (banner_w - (inner + 6)) // 2)

    def p(s):
        return f"{' ' * indent}{b}│{R}  {s}{' ' * (inner - wcswidth(_asi('', s)))}  {b}│{R}"

    print(f"{' ' * indent}{b}╭{'─' * (inner + 4)}╮{R}")
    print(p(label))
    print(p(""))
    for r in rows:
        print(p(r))
    print(p(""))
    print(p(tip))
    print(f"{' ' * indent}{b}╰{'─' * (inner + 4)}╯{R}")
    print()

    mcp = setup_mcp_server_standalone(logger)
    register_http_routes(mcp, logger)

    mcp.run(
        transport="http",
        host=host,
        port=port,
        show_banner=False,
    )
