#!/usr/bin/env python3
"""
HexStrike AI-PULSE — FastMCP 3.x Standalone Server (Phase 3)

🚀 Direct tool execution, no Flask, native HTTP/SSE transport
📊 Dashboard served from server_static/ via Starlette StaticFiles
"""

import os
import logging
from pathlib import Path

from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles

from mcp_core.server_setup import setup_mcp_server_standalone
from server_core.modern_visual_engine import ModernVisualEngine

# ============================================================================
# LOGGING CONFIGURATION (MUST BE FIRST)
# ============================================================================

from server_core.setup_logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Configuration
API_HOST = os.environ.get('HEXSTRIKE_HOST', '127.0.0.1')
API_PORT = int(os.environ.get('HEXSTRIKE_PORT', 8888))

if __name__ == "__main__":
    os.environ["FASTMCP_SHOW_SERVER_BANNER"] = "false"
    os.environ["FASTMCP_CHECK_FOR_UPDATES"] = "off"
    print(ModernVisualEngine.create_banner())
    logger.info("🚀 Starting HexStrike Pulse Standalone Server")
    logger.info(f"📡 Server: http://{API_HOST}:{API_PORT}")

    # Phase 3: Standalone MCP server with native HTTP transport
    mcp = setup_mcp_server_standalone(logger)

    # -------------------------------------------------------------------------
    # Dashboard — serve server_static/ assets + SPA index
    # index.html uses absolute paths (/assets/...) so we mount assets on /assets
    # and serve index.html on /dashboard
    # -------------------------------------------------------------------------
    static_dir = Path(__file__).parent / "server_static"
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

        # Serve /dashboard — SPA entry point
        @mcp.custom_route("/dashboard", methods=["GET"])
        async def dashboard_index(request):
            """Serve dashboard SPA index."""
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return Response("Dashboard not found", status_code=404)

        # Serve static root files (favicon, icons)
        @mcp.custom_route("/{filename:str}", methods=["GET"])
        async def root_static(request):
            """Serve root static files (favicon, icons)."""
            filename = request.path_params.get("filename", "")
            file_path = static_dir / filename
            if file_path.is_file() and file_path.suffix in (".svg", ".ico", ".png", ".txt"):
                return FileResponse(str(file_path))
            return Response("Not found", status_code=404)

        logger.info(f"📊 Dashboard at http://{API_HOST}:{API_PORT}/dashboard")
    else:
        logger.warning("⚠️ server_static/ not found — dashboard not available")

    # FastMCP 3.x native HTTP/SSE transport
    mcp.run(
        transport="http",
        host=API_HOST,
        port=API_PORT,
        log_level="warning",
    )
