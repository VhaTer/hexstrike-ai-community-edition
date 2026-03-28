from typing import Dict, Any
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context


def register_pywerview_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def pywerview(
        ctx: Context,
        request: str,
        target: str,
        username: str = "",
        password: str = "",
        dc_ip: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Run PywerView for Active Directory enumeration (PowerView-like).

        Args:
            request:         PywerView request type (get-netuser, get-netgroup,
                             get-netcomputer, get-netdomaincontroller...)
            target:          Target DC hostname or IP (-t)
            username:        Username (-u)
            password:        Password (-p)
            dc_ip:           Domain controller IP (--dc-ip)
            additional_args: Extra CLI arguments

        Returns:
            PywerView enumeration results
        """
        data = {
            "request":         request,
            "target":          target,
            "username":        username,
            "password":        password,
            "dc_ip":           dc_ip,
            "additional_args": additional_args,
        }

        await ctx.info(f"🔍 Starting pywerview {request} against {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("pywerview", data)
        )

        phases = [(50, "Enumerating AD objects...")]
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
            await ctx.info(f"✅ pywerview {request} completed")
        else:
            await ctx.error(f"❌ pywerview {request} failed: {result.get('error')}")
        return result
