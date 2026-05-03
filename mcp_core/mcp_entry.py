"""
mcp_core/mcp_entry.py

Track 2 refactor: Flask dependency removed.
HexStrikeClient and setup_mcp_server() (Flask-era) are gone.
Entry point now uses setup_mcp_server_standalone() directly.

hexstrike_mcp.py calls run_mcp() — this is the stdio transport path
used by Claude Desktop and other MCP clients that speak stdio.
"""

import sys
import logging
from mcp_core.server_setup import setup_mcp_server_standalone


def run_mcp(args, logger):
    """Run the HexStrike MCP server in standalone mode (stdio transport)."""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🔍 Debug logging enabled")

    logger.info("🚀 Starting HexStrike AI-PULSE MCP server (standalone)")

    try:
        mcp = setup_mcp_server_standalone(logger)
        logger.info("✅ HexStrike AI-PULSE MCP server ready")

        # stdio transport for Claude Desktop / MCP clients
        mcp.run(show_banner=False, log_level="WARNING")

    except Exception as e:
        logger.error(f"💥 Error starting MCP server: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
