from typing import Dict, Any
import asyncio
import mcp_core.osint_direct as _osint_direct
from fastmcp import Context


def register_osint_sherlock_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def sherlock(ctx: Context, username: str) -> Dict[str, Any]:
        """
        Execute Sherlock for username investigation across social networks.

        Args:
            username: The username to investigate

        Returns:
            Sherlock investigation results
        """
        data = {"username": username}

        await ctx.info(f"🔍 Starting Sherlock: {username}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _osint_direct.osint_exec("sherlock", data)
        )

        phases = [(33, "Searching social networks..."), (66, "Aggregating results...")]
        tick = 20
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=tick)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info(f"✅ Sherlock completed for {username}")
        else:
            await ctx.error(f"❌ Sherlock failed: {result.get('error')}")
        return result
