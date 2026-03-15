# mcp_tools/ops/system_monitoring.py

from typing import Dict, Any
from fastmcp import Context

def register_system_monitoring_tools(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def server_health(ctx: Context) -> Dict[str, Any]:
        """Check HexStrike server health, tool availability, and telemetry."""
        await ctx.info("🏥 Checking HexStrike server health")
        result = hexstrike_client.check_health()
        if result.get("status") == "healthy":
            await ctx.info(f"✅ Server healthy — {result.get('total_tools_available', 0)} tools available")
        else:
            await ctx.warning(f"⚠️ Server status: {result.get('status', 'unknown')}")
        return result

    @mcp.tool()
    async def get_cache_stats(ctx: Context) -> Dict[str, Any]:
        """Get cache performance statistics from the HexStrike server."""
        await ctx.info("💾 Getting cache statistics")
        result = hexstrike_client.safe_get("api/cache/stats")
        if "hit_rate" in result:
            await ctx.info(f"📊 Cache hit rate: {result.get('hit_rate')}")
        return result

    @mcp.tool()
    async def clear_cache(ctx: Context) -> Dict[str, Any]:
        """Clear the result cache on the HexStrike server."""
        await ctx.info("🧹 Clearing server cache")
        result = hexstrike_client.safe_post("api/cache/clear", {})
        if result.get("success"):
            await ctx.info("✅ Cache cleared")
        else:
            await ctx.error("❌ Failed to clear cache")
        return result

    @mcp.tool()
    async def get_telemetry(ctx: Context) -> Dict[str, Any]:
        """Get system performance and usage telemetry."""
        await ctx.info("📈 Getting system telemetry")
        result = hexstrike_client.safe_get("api/telemetry")
        if "commands_executed" in result:
            await ctx.info(f"📊 Commands executed: {result.get('commands_executed', 0)}")
        return result
