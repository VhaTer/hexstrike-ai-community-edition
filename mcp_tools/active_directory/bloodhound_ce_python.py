from typing import Dict, Any
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context


def register_bloodhound_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def bloodhound_python(
        ctx: Context,
        domain: str,
        username: str,
        password: str,
        dc_ip: str,
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Run bloodhound-python to collect BloodHound data from Active Directory.

        Args:
            domain:          Target AD domain (-d)
            username:        Username for authentication (-u)
            password:        Password for authentication (-p)
            dc_ip:           Domain controller IP (--dc)
            additional_args: Extra CLI arguments

        Returns:
            bloodhound-python collection results (ZIP file path in output)
        """
        data = {
            "domain":          domain,
            "username":        username,
            "password":        password,
            "dc_ip":           dc_ip,
            "additional_args": additional_args,
        }

        await ctx.info(f"🩸 Starting bloodhound-python against {domain} ({dc_ip})")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("bloodhound", data)
        )

        phases = [
            (25, "Collecting users..."),
            (50, "Collecting groups and computers..."),
            (75, "Collecting ACLs and GPOs..."),
        ]
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
            await ctx.info(f"✅ bloodhound-python completed for {domain}")
        else:
            await ctx.error(f"❌ bloodhound-python failed: {result.get('error')}")
        return result
