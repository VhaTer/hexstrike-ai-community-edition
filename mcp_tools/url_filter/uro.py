# mcp_tools/url_filter/uro.py

from fastmcp import Context
import mcp_core.misc_direct as _misc_direct
from typing import Dict, Any
import asyncio

def register_uro_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def uro_url_filtering(urls: str, whitelist: str = "", blacklist: str = "",
                         additional_args: str = "") -> Dict[str, Any]:
        """
        Execute uro for filtering out similar URLs.

        Args:
            urls: URLs to filter
            whitelist: Whitelist patterns
            blacklist: Blacklist patterns
            additional_args: Additional uro arguments

        Returns:
            Filtered URL results with duplicates removed
        """
        data = {
            "urls": urls,
            "whitelist": whitelist,
            "blacklist": blacklist,
            "additional_args": additional_args
        }
        await ctx.info("🔍 Starting uro URL filtering")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("uro", data)
        )
        if result.get("success"):
            await ctx.info("✅ uro URL filtering completed")
        else:
            await ctx.error("❌ uro URL filtering failed")
        return result
