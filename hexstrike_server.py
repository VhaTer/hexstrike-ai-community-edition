#!/usr/bin/env python3
"""
HexStrike AI-PULSE — FastMCP 3.x Standalone Server (Phase 3)

🚀 Direct tool execution, no Flask, native HTTP/SSE transport
📊 Dashboard served from server_static/ via Starlette StaticFiles
"""

import os
import logging
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime

from starlette.responses import FileResponse, Response, StreamingResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from mcp_core.server_setup import setup_mcp_server_standalone
from server_core.modern_visual_engine import ModernVisualEngine
from server_core.singletons import cache, telemetry, enhanced_process_manager
import server_core.config_core as config_core
from server_api.ops.system_monitoring import _get_tool_availability, _HEALTH_TOOL_CATEGORIES, _tool_availability_last_refresh

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
            "status": "healthy",
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
            "tool_availability_age_seconds": round(time.time() - _tool_availability_last_refresh, 1) if _tool_availability_last_refresh > 0 else None,

            # System resources
            "resources": current_usage,
            "resources_timestamp": datetime.now().isoformat(),

            # Cache stats
            "cache_stats": cache.get_stats(),
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

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        """HTTP health endpoint for IDEs, proxies, and external checks."""
        return _json_status_response(_build_dashboard_response())

    @mcp.custom_route("/ping", methods=["GET"])
    async def ping(request):
        """Minimal liveness endpoint."""
        return JSONResponse({"status": "ok", "server": "hexstrike-ai-pulse"})

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
                        # Keepalive if nothing new
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
    print(ModernVisualEngine.create_banner())
    logger.info("🚀 Starting HexStrike Pulse Standalone Server")
    logger.info(f"📡 Server: http://{API_HOST}:{API_PORT}")

    # Phase 3: Standalone MCP server with native HTTP transport
    mcp = setup_mcp_server_standalone(logger)

    register_http_routes(mcp, logger)

    # FastMCP 3.x native HTTP/SSE transport
    mcp.run(
        transport="http",
        host=API_HOST,
        port=API_PORT,
        show_banner=False,
    )
