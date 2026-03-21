# mcp_tools/waf_detect/wafw00f.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_recon_direct as _web_recon_direct

def register_wafw00f_tool(mcp, hexstrike_client, logger):

    @mcp.tool()
    async def wafw00f_scan(ctx: Context, target: str, additional_args: str = "") -> Dict[str, Any]:
        """
        Identify and fingerprint WAF products using wafw00f.

        Parameters:
        - target: Target URL or IP
        - additional_args: Additional wafw00f arguments
        """
        data = {"target": target, "additional_args": additional_args}
        await ctx.info(f"🛡️ Starting Wafw00f WAF detection: {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_recon_direct.web_recon_exec("wafw00f", data)
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
