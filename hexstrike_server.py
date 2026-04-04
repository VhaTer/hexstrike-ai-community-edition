#!/usr/bin/env python3
"""
HexStrike AI - FastMCP 3.x Standalone Server (Phase 3)

🚀 Direct tool execution, no Flask, native HTTP/SSE transport
📊 Dashboard served from server_static/ via Starlette StaticFiles mount
"""

import os
import logging
from pathlib import Path

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from mcp_core.server_setup import setup_mcp_server_standalone

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
    logger.info("🚀 Starting HexStrike FastMCP 3.x Standalone Server (Phase 3)")
    logger.info(f"📡 Server: http://{API_HOST}:{API_PORT}")
    logger.info("🔧 Direct tool execution - no Flask dependency")

    # Phase 3: Standalone MCP server with native HTTP transport
    mcp = setup_mcp_server_standalone(logger)

    # -------------------------------------------------------------------------
    # Dashboard — serve server_static/ as static files
    # -------------------------------------------------------------------------
    static_dir = Path(__file__).parent / "server_static"
    if static_dir.exists():
        @mcp.custom_route("/dashboard/{path:path}", methods=["GET"])
        async def dashboard_files(request):
            """Serve dashboard static files."""
            from starlette.responses import FileResponse, Response
            path = request.path_params.get("path", "")
            file_path = static_dir / path
            if file_path.is_file():
                return FileResponse(file_path)
            # Fallback to index.html for SPA routing
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(index)
            return Response("Not found", status_code=404)

        @mcp.custom_route("/dashboard", methods=["GET"])
        async def dashboard_index(request):
            """Serve dashboard index."""
            from starlette.responses import FileResponse, Response
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(index)
            return Response("Not found", status_code=404)

        logger.info(f"📊 Dashboard available at http://{API_HOST}:{API_PORT}/dashboard")
    else:
        logger.warning("⚠️ server_static/ not found — dashboard not available")

    # FastMCP 3.x native HTTP/SSE transport
    mcp.run(
        transport="http",
        host=API_HOST,
        port=API_PORT,
    )
