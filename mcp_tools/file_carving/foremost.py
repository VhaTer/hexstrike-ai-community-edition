# mcp_tools/file_carving/foremost.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.misc_direct as _misc_direct

def register_foremost_tool(mcp, hexstrike_client, logger):

    @mcp.tool()
    async def foremost_carving(ctx: Context, input_file: str, output_dir: str = "/tmp/foremost_output", file_types: str = "", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute Foremost for file carving with enhanced logging.

        Args:
            input_file: Input file or device to carve
            output_dir: Output directory for carved files
            file_types: File types to carve (jpg,gif,png,etc.)
            additional_args: Additional Foremost arguments

        Returns:
            File carving results
        """
        data = {
            "input_file": input_file,
            "output_dir": output_dir,
            "file_types": file_types,
            "additional_args": additional_args
        }
        await ctx.info(f"📁 Starting Foremost file carving: {input_file}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("foremost", data)
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
