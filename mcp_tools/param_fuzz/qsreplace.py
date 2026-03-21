# mcp_tools/param_fuzz/qsreplace.py

from fastmcp import Context
import mcp_core.misc_direct as _misc_direct
from typing import Dict, Any
import asyncio

def register_qsreplace_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def qsreplace_parameter_replacement(urls: str, replacement: str = "FUZZ",
                                       additional_args: str = "") -> Dict[str, Any]:
        """
        Execute qsreplace for query string parameter replacement.

        Args:
            urls: URLs to process
            replacement: Replacement string for parameters
            additional_args: Additional qsreplace arguments

        Returns:
            Parameter replacement results for fuzzing
        """
        data = {
            "urls": urls,
            "replacement": replacement,
            "additional_args": additional_args
        }
        await ctx.info("🔄 Starting qsreplace parameter replacement")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _misc_direct.misc_exec("qsreplace", data)
        )
        if result.get("success"):
            await ctx.info("✅ qsreplace parameter replacement completed")
        else:
            await ctx.error("❌ qsreplace parameter replacement failed")
        return result
