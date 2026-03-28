from typing import Dict, Any
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context


def register_adidnsdump_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def adidnsdump(
        ctx: Context,
        target: str,
        username: str = "",
        password: str = "",
        zone: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Run adidnsdump to enumerate DNS records from Active Directory via LDAP.

        Args:
            target:          Target DC hostname or IP
            username:        Optional username (-u)
            password:        Optional password (-p)
            zone:            Optional DNS zone to target (--zone)
            additional_args: Extra CLI arguments

        Returns:
            adidnsdump enumeration results
        """
        data = {
            "target":          target,
            "username":        username,
            "password":        password,
            "zone":            zone,
            "additional_args": additional_args,
        }

        await ctx.info(f"🔍 Starting adidnsdump against {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("adidnsdump", data)
        )

        phases = [(50, "Enumerating DNS records...")]
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
            await ctx.info(f"✅ adidnsdump completed for {target}")
        else:
            await ctx.error(f"❌ adidnsdump failed: {result.get('error')}")
        return result
