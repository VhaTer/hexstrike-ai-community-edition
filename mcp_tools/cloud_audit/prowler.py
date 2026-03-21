# mcp_tools/cloud_audit/prowler.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.security_direct as _security_direct

def register_prowler_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def prowler_scan(ctx: Context, provider: str = "aws", profile: str = "default", region: str = "", checks: str = "", output_dir: str = "/tmp/prowler_output", output_format: str = "json", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute Prowler for comprehensive cloud security assessment.

        Args:
            provider: Cloud provider (aws, azure, gcp)
            profile: AWS profile to use
            region: Specific region to scan
            checks: Specific checks to run
            output_dir: Directory to save results
            output_format: Output format (json, csv, html)
            additional_args: Additional Prowler arguments

        Returns:
            Cloud security assessment results
        """
        data = {
            "provider": provider,
            "profile": profile,
            "region": region,
            "checks": checks,
            "output_dir": output_dir,
            "output_format": output_format,
            "additional_args": additional_args
        }
        await ctx.info(f"☁️  Starting Prowler {provider} security assessment")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/tools/prowler", data)
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
