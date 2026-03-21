# mcp_tools/binary_analysis/angr.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_angr_tools(mcp, hexstrike_client, logger):
    
    @mcp.tool()
    async def angr_symbolic_execution(ctx: Context, binary: str, script_content: str = "",
                               find_address: str = "", avoid_addresses: str = "",
                               analysis_type: str = "symbolic", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute angr for symbolic execution and binary analysis.

        Args:
            binary: Binary to analyze
            script_content: Custom angr script content
            find_address: Address to find during symbolic execution
            avoid_addresses: Comma-separated addresses to avoid
            analysis_type: Type of analysis (symbolic, cfg, static)
            additional_args: Additional arguments

        Returns:
            Symbolic execution and binary analysis results
        """
        data = {
            "binary": binary,
            "script_content": script_content,
            "find_address": find_address,
            "avoid_addresses": avoid_addresses,
            "analysis_type": analysis_type,
            "additional_args": additional_args
        }
        await ctx.info(f"🔧 Starting angr analysis: {binary}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_postsafe_post("api/tools/angr", data)
        )

        done, _ = await asyncio.wait([future], timeout=30)
        if not done:
            await ctx.report_progress(50, 100)
            await ctx.info("⏳ Still running...")

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ Completed successfully")
        else:
            await ctx.error(f"❌ Failed: {result.get('error', 'unknown')}")
        return result
