# mcp_tools/web_crawl/hakrawler.py

from typing import Dict, Any
from fastmcp import Context

def register_hakrawler_tools(mcp, hexstrike_client, logger=None):

    @mcp.tool()
    async def hakrawler_crawl(
        ctx: Context,
        url: str,
        depth: int = 2,
        forms: bool = True,
        robots: bool = True,
        sitemap: bool = True,
        wayback: bool = False,
        additional_args: str = ""
    ) -> Dict[str, Any]:
        """
        Fast web endpoint discovery using hakrawler.

        Workflow position: quick crawl alongside or before katana.
        Hakrawler is faster and lighter than katana — good first pass
        to rapidly discover linked pages, forms, and standard paths.

        Note on parameter mapping (Kali hakrawler implementation):
        - url: piped via stdin (not -url flag)
        - depth: mapped to -d flag
        - forms: mapped to -s flag (show sources)
        - robots/sitemap/wayback: mapped to -subs for inclusion
        - always includes -u for unique URLs

        Parameters:
        - url: target URL (e.g. 'https://example.com')
        - depth: crawl depth — links to follow from root (default 2)
        - forms: include form action URLs in discovery (default True)
        - robots: parse and include robots.txt paths (default True)
        - sitemap: parse and include sitemap.xml URLs (default True)
        - wayback: include URLs from Wayback Machine (default False — use gau instead)
        - additional_args: extra hakrawler flags

        Prerequisites: target must be accessible.
        No JS rendering — static HTML crawling only.

        Output: list of discovered URLs including forms, robots, sitemap entries.

        hakrawler vs katana:
        - hakrawler — fast, lightweight, no JS rendering, good for traditional sites
        - katana    — slower, JS rendering, form extraction, better for modern SPAs

        Typical sequence:
            1. hakrawler_crawl(url='https://example.com')    — fast static crawl
            2. katana_crawl(url='https://example.com')       — deep JS crawl
            3. Deduplicate with anew
            4. Feed to gobuster/ffuf or injection tools
        """
        data = {
            "url": url,
            "depth": depth,
            "forms": forms,
            "robots": robots,
            "sitemap": sitemap,
            "wayback": wayback,
            "additional_args": additional_args
        }
        await ctx.info(f"🕷️ Starting hakrawler: {url}")
        result = hexstrike_client.safe_post("api/tools/hakrawler", data)
        if result.get("success"):
            await ctx.info(f"✅ hakrawler completed for {url}")
        else:
            await ctx.error(f"❌ hakrawler failed for {url}")
        return result
