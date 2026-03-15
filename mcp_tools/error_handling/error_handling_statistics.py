# mcp_tools/error_handling/error_handling_statistics.py

from typing import Dict, Any
from fastmcp import Context

def register_error_handling_statistics_tool(mcp, hexstrike_client, logger=None, HexStrikeColors=None):

    @mcp.tool()
    async def error_handling_statistics(ctx: Context) -> Dict[str, Any]:
        """
        Get error handling statistics and recent error patterns from the HexStrike server.

        Useful for debugging recurring tool failures or understanding
        which tools are failing most frequently.

        Output: total errors, error counts by type, recent error patterns,
        and recovery strategy success rates.
        """
        await ctx.info("📊 Retrieving error handling statistics")
        result = hexstrike_client.safe_get("api/error-handling/statistics")
        if result.get("success"):
            stats = result.get("statistics", {})
            total = stats.get("total_errors", 0)
            recent = stats.get("recent_errors_count", 0)
            await ctx.info(f"✅ Stats retrieved — total: {total}, recent: {recent}")
            error_counts = stats.get("error_counts_by_type", {})
            for error_type, count in error_counts.items():
                await ctx.info(f"  {error_type}: {count}")
        else:
            await ctx.error("❌ Failed to retrieve error statistics")
        return result
