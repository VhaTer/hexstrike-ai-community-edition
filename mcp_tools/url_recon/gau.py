# mcp_tools/url_recon/gau.py

from typing import Dict, Any
from fastmcp import Context

def register_gau_tool(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def gau_discovery(
        ctx: Context,
        domain: str,
        providers: str = "wayback,commoncrawl,otx,urlscan",
        include_subs: bool = True,
        blacklist: str = "png,jpg,gif,jpeg,swf,woff,svg,pdf,css,ico",
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fetch all known URLs for a domain from multiple archive sources using gau.

        Workflow position: passive URL discovery — use after subdomain enum,
        before active crawling or parameter discovery. Finds forgotten endpoints,
        old parameters, and legacy paths without touching the target directly.

        Parameters:
        - domain: target domain (e.g. 'example.com')
        - providers: comma-separated data sources (default: all major sources)
            'wayback'     — Wayback Machine (Internet Archive)
            'commoncrawl' — Common Crawl index
            'otx'         — AlienVault OTX
            'urlscan'     — urlscan.io
        - include_subs: include URLs from subdomains (default True)
        - blacklist: comma-separated extensions to exclude from results
        - additional_args: extra gau flags

        Prerequisites: none — purely passive, queries public archives.

        Output: list of historical URLs — may include endpoints no longer
        linked from the site but still live on the server.

        gau vs waybackurls:
        - gau          — queries 4 sources, more comprehensive
        - waybackurls  — Wayback Machine only, faster

        Typical URL recon pipeline:
            1. gau_discovery(domain='example.com')
            2. waybackurls_discovery(domain='example.com')
            3. Deduplicate with anew, filter with uro
            4. arjun_parameter_discovery on interesting endpoints
            5. dalfox/sqlmap on parameterized URLs
        """
        data = {
            "domain": domain,
            "providers": providers,
            "include_subs": include_subs,
            "blacklist": blacklist,
            "additional_args": additional_args
        }
        await ctx.info(f"📡 Starting gau URL discovery: {domain}")
        result = hexstrike_client.safe_post("api/tools/gau", data)
        if result.get("success"):
            await ctx.info(f"✅ gau completed for {domain}")
        else:
            await ctx.error(f"❌ gau failed for {domain}")
        return result
