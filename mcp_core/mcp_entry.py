"""
mcp_core/mcp_entry.py

Entry point for stdio transport used by Claude Desktop
and other MCP clients that speak stdio.

hexstrike_mcp.py calls run_mcp().
"""

import sys
import threading
import logging
from mcp_core.server_setup import setup_mcp_server_standalone


def _prewarm_singletons(logger):
    """Pre-initialize heavy singletons in background thread.

    Avoids first-access latency (~50-200ms for decision engine,
    ~5-20ms for tool stats store) when the user issues their first
    dashboard or planning request.
    """
    def _warm():
        logger.debug("🔍 Pre-warming lazy singletons...")
        try:
            from server_core.singletons import get_decision_engine, get_tool_stats_store
            get_decision_engine()
            get_tool_stats_store()
            logger.debug("✅ Pre-warming complete")
        except Exception as exc:
            logger.debug("⚠️ Pre-warming skipped: %s", exc)

    threading.Thread(target=_warm, daemon=True, name="prewarm").start()


def run_mcp(args, logger):
    """Run the HexStrike MCP server in standalone mode (stdio transport)."""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🔍 Debug logging enabled")

    logger.info("🚀 Starting HexStrike AI-PULSE MCP server (standalone)")

    try:
        mcp = setup_mcp_server_standalone(logger)
        from pulse_app import app as pulse_app
        mcp.add_provider(pulse_app)
        logger.info("✅ HexStrike AI-PULSE MCP server ready")

        # Pre-warm heavy singletons in background so the first user
        # request doesn't block on lazy initialization.
        _prewarm_singletons(logger)

        # stdio transport for Claude Desktop / MCP clients
        mcp.run(show_banner=False, log_level="WARNING")

    except Exception as e:
        logger.error(f"💥 Error starting MCP server: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
