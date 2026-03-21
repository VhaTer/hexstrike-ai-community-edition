# mcp_tools/iac_scan/checkov.py

from typing import Dict, Any
import asyncio
from fastmcp import Context

def register_checkov_tool(mcp, hexstrike_client, logger):
    
    @mcp.tool()
    async def checkov_iac_scan(ctx: Context, directory: str = ".", framework: str = "", check: str = "",
                        skip_check: str = "", output_format: str = "json",
                        additional_args: str = "") -> Dict[str, Any]:
        """
        Execute Checkov for infrastructure as code security scanning.

        Args:
            directory: Directory to scan
            framework: Framework to scan (terraform, cloudformation, kubernetes, etc.)
            check: Specific check to run
            skip_check: Check to skip
            output_format: Output format (json, yaml, cli)
            additional_args: Additional Checkov arguments

        Returns:
            Infrastructure as code security scanning results
        """
        data = {
            "directory": directory,
            "framework": framework,
            "check": check,
            "skip_check": skip_check,
            "output_format": output_format,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting Checkov IaC scan: {directory}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_postsafe_post("api/tools/checkov", data)
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
