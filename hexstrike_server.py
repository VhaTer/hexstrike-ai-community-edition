#!/usr/bin/env python3
"""
HexStrike AI - FastMCP 3.x Standalone Server (Phase 3)

🚀 Direct tool execution, no Flask, native HTTP/SSE transport
"""

import os
import logging
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
    
    # FastMCP 3.x native HTTP/SSE transport
    mcp.run(
        transport="http",
        host=API_HOST,
        port=API_PORT
    )
