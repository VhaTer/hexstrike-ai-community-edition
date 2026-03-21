# mcp_tools/binary_analysis/autopsy.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_autopsy_tools(mcp, hexstrike_client, logger):

    @mcp.tool()
    async def autopsy_analysis(ctx: Context) -> Dict[str, Any]:
        """
        Launch the Autopsy digital forensics web server and provide access instructions.

        Returns:
            dict: A dictionary containing connection details or error information for accessing the Autopsy web interface.
        """
        await ctx.info("🔍 Launching Autopsy web server")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/tools/binary_analysis/autopsy", {})
        )

        done, _ = await asyncio.wait([future], timeout=30)
        if not done:
            await ctx.report_progress(50, 100)
            await ctx.info("⏳ Still launching Autopsy...")

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ Autopsy web server launched")
            await ctx.info("💡 Connect via browser to the Autopsy web interface")
        else:
            await ctx.error(f"❌ Autopsy failed: {result.get('error', 'unknown')}")
        return result
