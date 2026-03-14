# mcp_tools/recon/subfinder.py

from typing import Dict, Any
from fastmcp import Context

def register_subfinder_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def subfinder_scan(
        ctx: Context,
        domain: str,
        silent: bool = True,
        all_sources: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fast passive subdomain discovery using subfinder.

        Workflow position: FIRST step in subdomain enumeration — low noise,
        no direct contact with target. Run before amass for a quick overview.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - silent: suppress banner and status output (default True — cleaner results)
        - all_sources: use all available passive sources (slower but more complete)
        - additional_args: extra subfinder flags
            '-o <file>'       — save output to file
            '-t <threads>'    — number of concurrent threads (default 10)
            '-timeout <sec>'  — timeout per source (default 30)

        Prerequisites: none — purely passive, queries public certificate logs,
        DNS datasets, and OSINT sources. Zero direct contact with target.

        Output: list of discovered subdomains (no IP resolution by default).
        Combine with httpx_probe to find live subdomains.

        subfinder vs amass:
        - subfinder — faster, passive only, best first step
        - amass     — slower, active options, graph analysis

        Typical sequence:
            1. subfinder_scan(domain='example.com', all_sources=True)
            2. amass_scan(domain='example.com', additional_args='-passive')
            3. httpx probe on combined results
        """
        data = {
            "domain": domain,
            "silent": silent,
            "all_sources": all_sources,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting subfinder: {domain}")
        result = hexstrike_client.safe_post("api/tools/subfinder", data)
        if result.get("success"):
            await ctx.info(f"✅ subfinder completed for {domain}")
        else:
            await ctx.error(f"❌ subfinder failed for {domain}")
        return result
