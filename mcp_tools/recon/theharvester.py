# mcp_tools/recon/theharvester.py

from typing import Dict, Any
import asyncio
from fastmcp import Context
import mcp_core.recon_direct as _recon_direct

def register_theharvester_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def theharvester_scan(
        ctx: Context,
        domain: str,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        OSINT gathering using theHarvester — emails, hosts, IPs from public sources.

        Workflow position: passive OSINT alongside subfinder/amass.
        Uniquely valuable for email harvesting and LinkedIn/social media intel
        that subdomain tools don't cover.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - additional_args: extra theHarvester flags
            '-b google,bing,linkedin'  — specific sources (default: all)
            '-l 500'                   — limit results per source (default 100)
            '-f <file>'                — save results to HTML/XML file
            '-s'                       — use Shodan (requires API key)
            '-v'                       — verify host names via DNS resolution

        Prerequisites: none — passive queries to public OSINT sources.
        Some sources require API keys (Shodan, Hunter.io, etc.) configured
        in ~/.theHarvester/api-keys.yaml.

        Output:
        - Email addresses (valuable for phishing/password spray)
        - Hostnames and subdomains
        - IP addresses and ASN info
        - LinkedIn profiles (if -b linkedin used)
        - Virtual hosts

        Typical recon sequence:
            1. subfinder_scan(domain='example.com')
            2. theharvester_scan(domain='example.com')    — emails + OSINT
            3. amass_scan(domain='example.com')
            4. httpx probe on all discovered hosts
        """
        data = {
            "domain": domain,
            "additional_args": additional_args
        }
        await ctx.info(f"🔍 Starting theHarvester: {domain}")
        await ctx.report_progress(0, 100)

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, lambda: _recon_direct.recon_exec("theharvester", data)
        )

        phases = [
            (20, "🌐 Querying search engines..."),
            (50, "📧 Harvesting emails and subdomains..."),
            (80, "📋 Processing results..."),
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
