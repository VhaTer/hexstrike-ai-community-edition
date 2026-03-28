from typing import Dict, Any
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context


def register_certipy_ad_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def certipy_ad(
        ctx: Context,
        action: str,
        target: str = "",
        username: str = "",
        password: str = "",
        dc_ip: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Run Certipy for Active Directory Certificate Services enumeration and attacks.

        Args:
            action:          Certipy action (find, req, auth, shadow, relay, forge...)
            target:          Target DC or domain (-target)
            username:        Username (-u)
            password:        Password (-p)
            dc_ip:           Domain controller IP (-dc-ip)
            additional_args: Extra CLI arguments

        Returns:
            Certipy execution results
        """
        data = {
            "action":          action,
            "target":          target,
            "username":        username,
            "password":        password,
            "dc_ip":           dc_ip,
            "additional_args": additional_args,
        }

        await ctx.info(f"🔍 Starting certipy {action}" + (f" against {target}" if target else ""))
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("certipy_ad", data)
        )

        phases = [(33, "Enumerating certificates..."), (66, "Processing results...")]
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
            await ctx.info(f"✅ certipy {action} completed")
        else:
            await ctx.error(f"❌ certipy {action} failed: {result.get('error')}")
        return result
