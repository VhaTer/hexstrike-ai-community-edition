from typing import Dict, Any
import asyncio
import mcp_core.osint_direct as _osint_direct
from fastmcp import Context


def register_osint_spiderfoot_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def spiderfoot(ctx: Context, target: str) -> Dict[str, Any]:
        """
        Execute SpiderFoot for OSINT automation.

        Args:
            target: The target domain or IP for SpiderFoot analysis

        Returns:
            SpiderFoot analysis results
        """
        data = {"target": target}

        await ctx.info(f"🔍 Starting SpiderFoot: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _osint_direct.osint_exec("spiderfoot", data)
        )

        phases = [(25, "Running modules..."), (50, "Correlating data..."), (75, "Building report...")]
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
            await ctx.info(f"✅ SpiderFoot completed for {target}")
        else:
            await ctx.error(f"❌ SpiderFoot failed: {result.get('error')}")
        return result
