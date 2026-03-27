from typing import Dict, Any
import asyncio
import mcp_core.osint_direct as _osint_direct
from fastmcp import Context


def register_osint_sublist3r_tool(mcp, hexstrike_client, logger):
    @mcp.tool()
    async def sublist3r(
        ctx: Context,
        domain: str,
        threads: int = 3,
        engine: str = "",
    ) -> Dict[str, Any]:
        """
        Execute Sublist3r for subdomain enumeration.

        Args:
            domain:  The target domain for subdomain enumeration
            threads: Number of threads to use (default: 3)
            engine:  Optional search engine to use (e.g. "google", "bing")

        Returns:
            Sublist3r enumeration results
        """
        data = {"domain": domain, "threads": threads, "engine": engine}

        await ctx.info(f"🔍 Starting Sublist3r: {domain} — {threads} threads")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _osint_direct.osint_exec("sublist3r", data)
        )

        phases = [(33, "Querying search engines..."), (66, "Enumerating subdomains...")]
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
            await ctx.info(f"✅ Sublist3r completed for {domain}")
        else:
            await ctx.error(f"❌ Sublist3r failed: {result.get('error')}")
        return result
