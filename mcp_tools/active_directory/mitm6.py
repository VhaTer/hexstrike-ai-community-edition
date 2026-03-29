from typing import Dict, Any
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context
from mcp_core.elicitation import confirm_destructive_action


def register_mitm6_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def mitm6(
        ctx: Context,
        interface: str,
        domain: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Run mitm6 for IPv6 DNS takeover attacks against Active Directory.

        Args:
            interface:       Network interface to listen on (-i)
            domain:          Target AD domain to filter (-d)
            additional_args: Extra CLI arguments

        Returns:
            mitm6 execution results
        """
        data = {
            "interface":       interface,
            "domain":          domain,
            "additional_args": additional_args,
        }

        confirmed = await confirm_destructive_action(
            ctx,
            action=f"mitm6 IPv6 DNS takeover on {interface}",
            detail=f"Domain: {domain}" if domain else "All domains",
            warning="This poisons IPv6 DNS for all hosts on the network segment.",
        )
        if not confirmed:
            return {"success": False, "error": "Cancelled by user — mitm6 not started"}

        await ctx.info(f"🔴 Starting mitm6 on {interface}" + (f" targeting {domain}" if domain else ""))
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("mitm6", data)
        )

        phases = [(33, "Setting up IPv6..."), (66, "Poisoning DNS...")]
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
            await ctx.info("✅ mitm6 completed")
        else:
            await ctx.error(f"❌ mitm6 failed: {result.get('error')}")
        return result
