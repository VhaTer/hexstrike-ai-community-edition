# mcp_tools/iac_scan/terrascan.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.security_direct as _security_direct

def register_terrascan_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def terrascan_iac_scan(ctx: Context, scan_type: str = "all", iac_dir: str = ".",
                          policy_type: str = "", output_format: str = "json",
                          severity: str = "", additional_args: str = "") -> Dict[str, Any]:
        """
        Execute Terrascan for infrastructure as code security scanning.

        Args:
            scan_type: Type of scan (all, terraform, k8s, etc.)
            iac_dir: Infrastructure as code directory
            policy_type: Policy type to use
            output_format: Output format (json, yaml, xml)
            severity: Severity filter (high, medium, low)
            additional_args: Additional Terrascan arguments

        Returns:
            Infrastructure as code security scanning results
        """
        data = {
            "scan_type": scan_type,
            "iac_dir": iac_dir,
            "policy_type": policy_type,
            "output_format": output_format,
            "severity": severity,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting Terrascan IaC scan: {iac_dir}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: hexstrike_client.safe_post("api/tools/terrascan", data)
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
