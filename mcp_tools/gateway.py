# mcp_tools/gateway.py
# Track 2 refactor: Flask gateway removed.
# classify_task() called hexstrike_client.safe_post() → Flask — deleted.
# run_tool() Flask fallback removed — all routes go through *_direct.py modules.
# register_gateway_tools() kept as a no-op shim for any legacy callers still
# importing it. Remove this file entirely once mcp_entry.py / hexstrike_mcp.py
# are confirmed unused.

from typing import Dict, Any
import json
import asyncio


def register_gateway_tools(mcp, hexstrike_client=None):
    """
    Legacy shim — previously registered classify_task and run_tool on a Flask-backed
    FastMCP instance. Now a no-op: all tool execution goes through
    setup_mcp_server_standalone() and run_security_tool() directly.
    """
    pass
