# mcp_tools/ops/process_management.py

from typing import Dict, Any
from fastmcp import Context

def register_process_management_tools(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def list_active_processes(ctx: Context) -> Dict[str, Any]:
        """List all active tool processes on the HexStrike server."""
        await ctx.info("📊 Listing active processes")
        result = hexstrike_client.safe_get("api/processes/list")
        if result.get("success"):
            await ctx.info(f"✅ Found {result.get('total_count', 0)} active processes")
        else:
            await ctx.error("❌ Failed to list processes")
        return result

    @mcp.tool()
    async def get_process_status(ctx: Context, pid: int) -> Dict[str, Any]:
        """Get status of a specific process by PID."""
        await ctx.info(f"🔍 Checking process {pid}")
        result = hexstrike_client.safe_get(f"api/processes/status/{pid}")
        if result.get("success"):
            await ctx.info(f"✅ Process {pid} status retrieved")
        else:
            await ctx.error(f"❌ Process {pid} not found")
        return result

    @mcp.tool()
    async def terminate_process(ctx: Context, pid: int) -> Dict[str, Any]:
        """Terminate a running process by PID."""
        await ctx.info(f"🛑 Terminating process {pid}")
        result = hexstrike_client.safe_post(f"api/processes/terminate/{pid}", {})
        if result.get("success"):
            await ctx.info(f"✅ Process {pid} terminated")
        else:
            await ctx.error(f"❌ Failed to terminate process {pid}")
        return result

    @mcp.tool()
    async def pause_process(ctx: Context, pid: int) -> Dict[str, Any]:
        """Pause a running process by PID."""
        await ctx.info(f"⏸️ Pausing process {pid}")
        result = hexstrike_client.safe_post(f"api/processes/pause/{pid}", {})
        if result.get("success"):
            await ctx.info(f"✅ Process {pid} paused")
        else:
            await ctx.error(f"❌ Failed to pause process {pid}")
        return result
