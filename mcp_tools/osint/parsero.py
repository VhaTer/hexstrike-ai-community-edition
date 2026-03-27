from typing import Dict, Any
import asyncio
import mcp_core.osint_direct as _osint_direct
from fastmcp import Context


def register_osint_parsero_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def parsero(
        ctx: Context,
        target: str,
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Execute Parsero for Robots.txt analysis.

        Args:
            target:          The target URL for Parsero analysis
            additional_args: Optional additional arguments for Parsero

        Returns:
            Parsero analysis results
        """
        data = {"target": target, "additional_args": additional_args}

        await ctx.info(f"🔍 Starting Parsero: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _osint_direct.osint_exec("parsero", data)
        )

        phases = [(50, "Parsing robots.txt entries...")]
        tick = 15
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=tick)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info(f"✅ Parsero completed for {target}")
        else:
            await ctx.error(f"❌ Parsero failed: {result.get('error')}")
        return result
