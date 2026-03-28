from typing import Any, Dict, Optional
import asyncio
import mcp_core.active_directory_direct as _ad_direct
from fastmcp import Context


def register_impacket(mcp, hexstrike_client, logger, HexStrikeColors=None):
    @mcp.tool()
    async def impacket_run(
        ctx: Context,
        script: str,
        target: str,
        options: Optional[Dict[str, Any]] = None,
        extra_args: str = "",
    ) -> Dict[str, Any]:
        """
        Execute any Impacket script directly (no Flask).

        Args:
            script:     Impacket script name without 'impacket-' prefix
                        (e.g. GetADUsers, GetNPUsers, psexec, smbclient)
            target:     Primary target/credential string
            options:    Dict of CLI flags e.g. {"dc-ip": "10.10.10.1", "all": True}
            extra_args: Raw extra CLI args for edge cases

        Returns:
            Execution result
        """
        data = {
            "script":    script,
            "target":    target,
            "options":   options or {},
            "extra_args": extra_args,
        }

        await ctx.info(f"🧨 Starting impacket-{script} against {target}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _ad_direct.ad_exec("impacket", data)
        )

        phases = [(33, "Running script..."), (66, "Processing output...")]
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
            await ctx.info(f"✅ impacket-{script} completed")
        else:
            await ctx.error(f"❌ impacket-{script} failed: {result.get('error')}")
        return result
