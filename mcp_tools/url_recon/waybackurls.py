# mcp_tools/url_recon/waybackurls.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.web_recon_direct as _web_recon_direct

def register_waybackurls_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def waybackurls_discovery(
        ctx: Context,
        domain: str,
        get_versions: bool = False,
        no_subs: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fetch historical URLs from the Wayback Machine (Internet Archive).

        Workflow position: passive URL discovery alongside gau.
        Faster than gau — single source, ideal for quick reconnaissance
        or when other sources are rate-limited.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - get_versions: fetch all saved versions of each URL
                        (useful for finding old API versions or removed content)
        - no_subs: exclude subdomain URLs (focus on main domain only)
        - additional_args: extra waybackurls flags

        Prerequisites: none — purely passive Wayback Machine queries.

        Output: list of historical URLs archived by the Wayback Machine.
        Particularly useful for:
        - Finding old API endpoints (v1, v2, legacy)
        - Discovering backup files (.bak, .old, .zip)
        - Identifying removed but potentially still-live admin panels
        - Finding parameters in old URLs for injection testing

        Typical pipeline:
            1. gau_discovery(domain='example.com')          — multi-source
            2. waybackurls_discovery(domain='example.com')  — Wayback only
            3. Combine + deduplicate results with anew
            4. Filter interesting URLs (with params) for testing
        """
        data = {
            "domain": domain,
            "get_versions": get_versions,
            "no_subs": no_subs,
            "additional_args": additional_args
        }
        await ctx.info(f"🕰️ Starting waybackurls discovery: {domain}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _web_recon_direct.web_recon_exec("waybackurls", data)
        )

        phases = [
            (30, "📚 Querying Wayback Machine..."),
            (65, "📋 Processing historical URLs..."),
            (90, "🔍 Filtering results..."),
        ]
        for progress, message in phases:
            done, _ = await asyncio.wait([future], timeout=10)
            if done:
                break
            await ctx.report_progress(progress, 100)
            await ctx.info(message)

        result = await future
        await ctx.report_progress(100, 100)

        if result.get("success"):
            await ctx.info("✅ Completed successfully")
        else:
            await ctx.error(f"❌ Failed: {result.get('error', 'unknown')}")
        return result
